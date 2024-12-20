from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import gc
import os

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IS_BEYOND_ACCURACY_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL #  needed for article age
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import (
    slice_join_dataframes,
    concat_str_columns,
    #chunk_dataframe,
    split_df_chunks,
    split_df_fraction,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# conda activate ./venv/
# python -i examples/00_quick_start/nrms_ebnerd.py


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    ) 
    return df_behaviors

# TODO: CHANGE THIS
PATH = Path("/dtu/blackhole/16/213878/ebnerd_data").expanduser()
DUMP_DIR = Path("ebnerd_predictions").resolve()
DUMP_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42

MODEL_NAME = f"NRMS-{dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}-{SEED}"
# MODEL_NAME = "NRMS-382861963-2024-11-12 01:34:49.050070"

MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
TEST_DF_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)

print(f"Dir: {MODEL_NAME}")

DATASPLIT = "ebnerd_small"
MAX_TITLE_LENGTH = 30

HISTORY_SIZE = 20
FRACTION = 0.001
EPOCHS = 1
FRACTION_TEST = 0.001
#
hparams_nrms.history_size = HISTORY_SIZE

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
BATCH_SIZE_TEST_WO_B = 16
BATCH_SIZE_TEST_W_B = 1
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL # needed for article age
]

df = (
    pl.concat(
        [
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "train"),
                history_size=HISTORY_SIZE,
            ),
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "validation"),
                history_size=HISTORY_SIZE,
            ),
        ]
    )
    .sample(fraction=FRACTION, shuffle=True, seed=SEED)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
)

last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

# TODO: CHANGE THIS
df_articles = pl.read_parquet(PATH.joinpath("larger_articles.parquet"))

# =>
TRANSFORMER_MODEL_NAME = "meta-llama/Llama-2-7b-hf" # changed
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]

# LOAD HUGGINGFACE:
transformer_model = AutoModelForCausalLM.from_pretrained(
    TRANSFORMER_MODEL_NAME,
    token=True,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto",  # Automatically handle model splitting across GPUs
    offload_folder="offload_to_cpu",
)

transformer_tokenizer = AutoTokenizer.from_pretrained(
    TRANSFORMER_MODEL_NAME,
)

if transformer_tokenizer.pad_token is None:
    transformer_tokenizer.pad_token = transformer_tokenizer.eos_token

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

#=====================Weight Calculation=================================


def calculate_article_age(df, df_articles, max_value=1_000_000):
    """
    Add a new column to df_train containing normalized time deltas for all in-view articles.
    Negative deltas are set to 0, deltas are capped at max_value, and then normalized to [0, 1].
    
    Args:
        df_train (pl.DataFrame): DataFrame containing impression_time and inview_article_ids.
        df_articles (pl.DataFrame): DataFrame containing article_id and published_time.
        max_value (float): Maximum value for normalization (default is 1,000,000).

    Returns:
        pl.DataFrame: Updated df_train with a new column `time_deltas_normalized`.
    """
    # Create a mapping of article_id to published_time
    article_time_mapping = dict(
        zip(
            df_articles["article_id"].to_list(),
            df_articles["published_time"].to_list(),
        )
    )

    # Define a helper function to calculate and normalize deltas
    def compute_and_normalize_time_deltas(inview_article_ids, impression_time):
        deltas = []
        for article_id in inview_article_ids:
            published_time = article_time_mapping.get(article_id)
            if published_time:
                # Calculate delta in minutes
                delta = (impression_time - published_time).total_seconds() / 60
                # Set negative deltas to 0, cap at max_value, and normalize
                delta = min(max(delta, 0), max_value)
                normalized_delta = delta / max_value  # Normalize to [0, 1]
                deltas.append(normalized_delta)
            else:
                deltas.append(None)  # Handle missing published_time
        return deltas

    # Apply the function to compute and normalize time deltas
    df = df.with_columns(
        pl.struct(["article_ids_inview", "impression_time"]).apply(
            lambda row: compute_and_normalize_time_deltas(
                row["article_ids_inview"], row["impression_time"]
            )
        ).alias("article_age_normalized")
    )

    return df

# Call the function
df_train = calculate_article_age(df_train, df_articles)
df_validation = calculate_article_age(df_validation, df_articles)
#======================================================

########### Image embeddings

# Target embedding size (same as text embedding size)
TARGET_EMBEDDING_SIZE = 30
INPUT_DIM = 1024

# # Load data
df_image_embeddings = pl.read_parquet(PATH.joinpath("image_embeddings.parquet"))

########## NEW SVD

# Load and preprocess image embeddings
valid_embeddings = [
    emb for emb in df_image_embeddings["image_embedding"].to_list()
    if emb is not None and len(emb) == 1024 and not all(v == 0.0 for v in emb)
]

# Convert to NumPy array
valid_embeddings = np.array(valid_embeddings, dtype=np.float32)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=30, random_state=42)  # Reduce to 30 dimensions
reduced_embeddings = svd.fit_transform(valid_embeddings)

# print(f"Original shape: {valid_embeddings.shape}")
# print(f"Reduced shape: {reduced_embeddings.shape}")

# Add reduced embeddings back to the DataFrame
# Use None for entries that were not processed in SVD
reduced_embeddings_list = []
embedding_index = 0
for emb in df_image_embeddings["image_embedding"]:
    if emb is not None and len(emb) == 1024 and not all(v == 0.0 for v in emb):
        reduced_embeddings_list.append(reduced_embeddings[embedding_index].tolist())
        embedding_index += 1
    else:
        reduced_embeddings_list.append([0.0] * 30)  # Replace missing embeddings with zeros

df_image_embeddings = df_image_embeddings.with_columns(
    pl.Series(name="reduced_image_embedding", values=reduced_embeddings_list)
)

# Combine text and reduced image embeddings
df_articles = df_articles.join(df_image_embeddings, on="article_id", how="left")

df_articles = df_articles.with_columns(
    pl.when(pl.col("reduced_image_embedding").is_null())
    .then([0.0] * 30)  # Ensure missing embeddings are replaced with a zero vector of length 30
    .otherwise(pl.col("reduced_image_embedding"))
    .alias("reduced_image_embedding")
)

# Updated combine_embeddings function
def combine_embeddings(text_embedding, image_embedding):
    """
    Combines text and image embeddings, ensuring image_embedding is reduced or zeroed.
    """
    text_embedding = [float(x) for x in text_embedding]
    image_embedding = [float(x) for x in image_embedding]  # Ensure consistency in data types
    assert len(text_embedding) == 30, f"Text embedding size mismatch: {len(text_embedding)}"
    assert len(image_embedding) == 30, f"Image embedding size mismatch: {len(image_embedding)}"
    return text_embedding + image_embedding

print("~~~~~~~~~~~articles_schema_", df_articles.schema)
df_articles = df_articles.with_columns(
    pl.struct([token_col_title, "reduced_image_embedding"])
    .apply(lambda row: combine_embeddings(row[token_col_title], row["reduced_image_embedding"]))
    .alias("combined_embedding")
)


# Image embeddings
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col="combined_embedding"
)

# =>
print("Init train- and val-dataloader")
# Image embeddings
train_dataloader = NRMSDataLoaderPretransform(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE_TRAIN,
)
val_dataloader = NRMSDataLoaderPretransform(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE_VAL,
)

####################

# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1
)

model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)

hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, early_stopping],
)

# Process the predictions
val_dataloader.eval_mode=True
scores = model.scorer.predict(val_dataloader)
df_validation = add_prediction_scores(df_validation, scores.tolist()).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())
clear_session()

del (
    transformer_model,
    train_dataloader,
    val_dataloader,
    df_validation,
    df_train,
)
gc.collect()

print(f"saving model: {MODEL_WEIGHTS}")
model.model.save_weights(MODEL_WEIGHTS)
print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_weights(MODEL_WEIGHTS)

# =====================================================================================
print("Initiating testset...")
df_test = (
    ebnerd_from_path(
        PATH.joinpath("ebnerd_testset", "test"),
        history_size=HISTORY_SIZE,
    ).pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=HISTORY_SIZE,
            padding_value=0,
            enable_warning=False,
        )
    .sample(fraction=FRACTION_TEST)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.first()
        .alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element() * 0)
        .alias(DEFAULT_LABELS_COL)
    )
)

df_test_articles = pl.read_parquet(PATH.joinpath("ebnerd_testset", "articles.parquet"))
df_test_articles, cat_cal = concat_str_columns(df_test_articles, columns=TEXT_COLUMNS_TO_USE)
df_test_articles, token_col_title = convert_text2encoding_with_transformers(
    df_test_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)
df_test_articles = df_test_articles.join(df_image_embeddings, on="article_id", how="left")
df_test_articles = df_test_articles.with_columns(
    pl.when(pl.col("reduced_image_embedding").is_null())
    .then([0.0] * 30)  # Ensure missing embeddings are replaced with a zero vector of length 30
    .otherwise(pl.col("reduced_image_embedding"))
    .alias("reduced_image_embedding")
)

df_test_articles = df_test_articles.with_columns(
    pl.struct([token_col_title, "reduced_image_embedding"])
    .apply(lambda row: combine_embeddings(row[token_col_title], row["reduced_image_embedding"]))
    .alias("combined_embedding")
)
article_mapping = create_article_id_to_value_mapping(
    df=df_test_articles, value_col="combined_embedding"
)
df_test = calculate_article_age(df_test, df_articles)
# Split test in beyond-accuracy TRUE / FALSE. In the BA 'article_ids_inview' is 250.
df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
# Evaluating the test test can be memory intensive, we'll chunk it up:
TEST_CHUNKS_DIR = ARTIFACT_DIR.joinpath("test_chunks")
TEST_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
df_pred_test_wo_beyond = []
print("Initiating testset without beyond-accuracy...")
for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
    print(f"Test chunk: {i}/{len(df_test_chunks)}")
    # Initialize DataLoader
    test_dataloader_wo_b = NRMSDataLoader(
        behaviors=df_test_chunk,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE_TEST_WO_B,
    )
    # Predict and clear session
    scores = model.scorer.predict(test_dataloader_wo_b)
    tf.keras.backend.clear_session()

    # Process the predictions
    df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )

    # Save the processed chunk
    df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_CHUNKS_DIR.joinpath(f"pred_wo_ba_{i}.parquet")
    )

    # Append and clean up
    df_pred_test_wo_beyond.append(df_test_chunk)

    # Cleanup
    del df_test_chunk, test_dataloader_wo_b, scores
    gc.collect()

df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_wo_ba.parquet")
)
# =====================================================================================
print("Initiating testset with beyond-accuracy...")
test_dataloader_w_b = NRMSDataLoader(
    behaviors=df_test_w_beyond,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_W_B,
)
scores = model.scorer.predict(test_dataloader_w_b)
df_pred_test_w_beyond = add_prediction_scores(
    df_test_w_beyond, scores.tolist()
).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)
df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_w_ba.parquet")
)

# =====================================================================================
print("Saving prediction results...")
df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    ARTIFACT_DIR.joinpath("test_predictions.parquet")
)
import shutil
if TEST_CHUNKS_DIR.exists() and TEST_CHUNKS_DIR.is_dir():
    shutil.rmtree(TEST_CHUNKS_DIR)

write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=ARTIFACT_DIR.joinpath("predictions.txt"),
    filename_zip=f"{MODEL_NAME}-{SEED}-{DATASPLIT}.zip",
)
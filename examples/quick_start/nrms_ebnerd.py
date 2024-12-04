from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import torch
import gc
import os
from tqdm import tqdm

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


PATH = Path("/dtu/blackhole/0c/215532/ebnerd_data").expanduser()
DUMP_DIR = Path("ebnerd_predictions").resolve()
DUMP_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42

MODEL_NAME = f"NRMS-{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-{SEED}"
# MODEL_NAME = "NRMS-382861963-2024-11-12 01:34:49.050070"

MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
TEST_DF_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)

print(f"Dir: {MODEL_NAME}")

DATASPLIT = "ebnerd_small"
MAX_TITLE_LENGTH = 30

# TODO: MAX_ABSTRACT_LENGTH = 50
HISTORY_SIZE = 20
FRACTION = 0.05
EPOCHS = 10
FRACTION_TEST = 0.05
#
hparams_nrms.history_size = HISTORY_SIZE

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
BATCH_SIZE_TEST_WO_B = 16
BATCH_SIZE_TEST_W_B = 2
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
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
df_validation = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
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
#df_train, df_validation = split_df_fraction(df_train, fraction=0.9, seed=SEED, shuffle=False)

# df_test = df_validation
# df_train = df_train[:100]
# df_validation = df_validation[:100]
# df_test = df_test[:100]
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

# Load LLaMA model and tokenizer
llama_model_name = "meta-llama/Llama-3.2-1b"

llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
# Set padding token
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Use eos_token as pad_token

llama_model = AutoModel.from_pretrained(llama_model_name)

def create_llama_embeddings(df_articles, text_columns, max_length):
    embeddings = []
    for row in tqdm(df_articles.iter_rows(named=True), total=len(df_articles), desc="Creating LLaMA embeddings"):
        # Concatenate title and subtitle
        text = " ".join([str(row[col]) for col in text_columns if row[col] is not None])
        inputs = llama_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = llama_model(**inputs)
            # Use mean pooling of last hidden states as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

# # Create LLaMA embeddings for articles
# TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
# llama_embeddings = create_llama_embeddings(df_articles, TEXT_COLUMNS_TO_USE, MAX_TITLE_LENGTH)

# # Store embedding dimension for model configuration
# EMBEDDING_DIM = llama_embeddings.shape[1]

# # Update article mapping to use LLaMA embeddings
# article_mapping = {
#     article_id: embedding 
#     for article_id, embedding in zip(
#         df_articles['article_id'].to_numpy(), 
#         llama_embeddings
#     )
# }

TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
# Extract required article IDs from behaviors
required_article_ids = set(
    df_train[DEFAULT_HISTORY_ARTICLE_ID_COL].explode().to_list() +
    df_train[DEFAULT_INVIEW_ARTICLES_COL].explode().to_list() +
    df_validation[DEFAULT_HISTORY_ARTICLE_ID_COL].explode().to_list() +
    df_validation[DEFAULT_INVIEW_ARTICLES_COL].explode().to_list()
)

# Filter df_articles to only include required articles
df_articles_filtered = df_articles.filter(pl.col('article_id').is_in(required_article_ids))

# Create LLaMA embeddings for filtered articles
llama_embeddings = create_llama_embeddings(
    df_articles=df_articles_filtered,
    text_columns=TEXT_COLUMNS_TO_USE,
    max_length=MAX_TITLE_LENGTH
)

# Store embedding dimension for model configuration
EMBEDDING_DIM = llama_embeddings.shape[1]

# Update article mapping to use LLaMA embeddings
article_mapping = {
    article_id: embedding 
    for article_id, embedding in zip(
        df_articles['article_id'].to_numpy(), 
        llama_embeddings
    )
}

# Clean up LLaMA resources
del llama_model, llama_tokenizer
torch.cuda.empty_cache()
gc.collect()

# Update model hyperparameters
hparams_nrms.embedding_dim = EMBEDDING_DIM

# Create and train model
model = NRMSModel(
    hparams=hparams_nrms,
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
    llama_model,
    llama_tokenizer,
    train_dataloader,
    val_dataloader,
    df_validation,
    df_train,
)

# After generating embeddings
del llama_model, llama_tokenizer
torch.cuda.empty_cache()  # If using GPU
gc.collect()

print(f"saving model: {MODEL_WEIGHTS}")
model.model.save_weights(MODEL_WEIGHTS)
print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_weights(MODEL_WEIGHTS)

# =>
# print("Init df_test")
# df_test = (
#     ebnerd_from_path(PATH.joinpath("ebnerd_testset", "test"), history_size=HISTORY_SIZE)
#     .sample(fraction=FRACTION_TEST)
#     .with_columns(
#         pl.col(DEFAULT_INVIEW_ARTICLES_COL)
#         .list.first()
#         .alias(DEFAULT_CLICKED_ARTICLES_COL)
#     )
#     .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
#     .with_columns(
#         pl.col(DEFAULT_INVIEW_ARTICLES_COL)
#         .list.eval(pl.element() * 0)
#         .alias(DEFAULT_LABELS_COL)
#     )
# )
# # Split test in beyond-accuracy. BA samples have more 'article_ids_inview'.
# df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
# df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

# df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
# df_pred_test_wo_beyond = []

# for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
#     print(f"Init test-dataloader: {i}/{len(df_test_chunks)}")
#     # Initialize DataLoader
#     test_dataloader_wo_b = NRMSDataLoader(
#         behaviors=df_test_chunk,
#         article_dict=article_mapping,
#         unknown_representation="zeros",
#         history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#         eval_mode=True,
#         batch_size=BATCH_SIZE_TEST_WO_B,
#     )
#     # Predict and clear session
#     scores = model.scorer.predict(test_dataloader_wo_b)
#     clear_session()

#     # Process the predictions
#     df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
#         pl.col("scores")
#         .map_elements(lambda x: list(rank_predictions_by_score(x)))
#         .alias("ranked_scores")
#     )

#     # Save the processed chunk
#     df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
#         TEST_DF_DUMP.joinpath(f"pred_wo_ba_{i}.parquet")
#     )

#     # Append and clean up
#     df_pred_test_wo_beyond.append(df_test_chunk)

#     # Cleanup
#     del df_test_chunk, test_dataloader_wo_b, scores
#     gc.collect()

# # =>
# df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
# df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
#     TEST_DF_DUMP.joinpath("pred_wo_ba.parquet")
# )

# print("Init test-dataloader: beyond-accuracy")
# test_dataloader_w_b = NRMSDataLoader(
#     behaviors=df_test_w_beyond,
#     article_dict=article_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     eval_mode=True,
#     batch_size=BATCH_SIZE_TEST_W_B,
# )
# scores = model.scorer.predict(test_dataloader_w_b)
# df_pred_test_w_beyond = add_prediction_scores(
#     df_test_w_beyond, scores.tolist()
# ).with_columns(
#     pl.col("scores")
#     .map_elements(lambda x: list(rank_predictions_by_score(x)))
#     .alias("ranked_scores")
# )
# df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
#     TEST_DF_DUMP.joinpath("pred_w_ba.parquet")
# )

# # =>
# df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
# df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
#     TEST_DF_DUMP.joinpath("pred_concat.parquet")
# )
# metrics = MetricEvaluator(
#     labels=df_validation["labels"].to_list(),
#     predictions=df_validation["scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# metrics.evaluate()

# write_submission_file(
#     impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
#     prediction_scores=df_test["ranked_scores"],
#     path=DUMP_DIR.joinpath("predictions.txt"),
#     filename_zip=f"{DATASPLIT}_predictions-{MODEL_NAME}.zip",
# )



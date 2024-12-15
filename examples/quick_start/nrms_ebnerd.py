from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import torch
import gc
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

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
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
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


PATH = Path("../../../ebnerd_data").expanduser()
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
FRACTION = 0.001
EPOCHS = 10
FRACTION_TEST = 0.001
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
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
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

print("Available columns:", df.columns)


last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

# =>
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)

########### Image embeddings

# Target embedding size (same as text embedding size)
TARGET_EMBEDDING_SIZE = 30
# INPUT_DIM = 1024 + 30  # Combined text and image embedding size

# # Load data
df_image_embeddings = pl.read_parquet(PATH.joinpath("image_embeddings.parquet"))

# # Combine and reduce embeddings
# def train_combined_linear_transform_model(embedding_dataset, input_dim=1054, target_dim=30):
#     # Input layer
#     input_layer = Input(shape=(input_dim,))
#     # Linear transformation layer
#     output_layer = Dense(target_dim, activation=None)(input_layer)

#     # Create the model
#     model = Model(inputs=input_layer, outputs=output_layer)

#     # Compile the model
#     model.compile(optimizer='adam', loss='mse')

#     # Train on combined embeddings
#     model.fit(embedding_dataset, np.zeros((len(embedding_dataset), target_dim)), epochs=10, batch_size=32)

#     return model

# # Function to combine and reduce embeddings
# def combine_and_reduce_embeddings(text_embedding, image_embedding):
#     # Convert text_embedding to float
#     text_embedding = [float(x) for x in text_embedding]

#     # Ensure valid embedding for image
#     if len(image_embedding) != 1024:
#         image_embedding = [0.0] * 1024

#     # Concatenate text and image embeddings
#     combined_embedding = text_embedding + list(image_embedding)

#     # Convert to tensor and reduce dimensionality
#     reduced_embedding = combined_transform_model.predict(np.array([combined_embedding]), verbose=0)
#     return reduced_embedding[0]

# # Prepare the combined embedding dataset for training
# valid_embeddings = [
#     text_embedding + list(image_embedding)
#     for text_embedding, image_embedding in zip(df_articles[token_col_title].to_list(), df_image_embeddings["image_embedding"].to_list())
#     if text_embedding is not None and len(text_embedding) == 30
#     and image_embedding is not None and len(image_embedding) == 1024 and not all(v == 0.0 for v in image_embedding)
# ]
# combined_embedding_dataset = np.array(valid_embeddings)

# # Train the combined transformation model
# combined_transform_model = train_combined_linear_transform_model(combined_embedding_dataset, INPUT_DIM, TARGET_EMBEDDING_SIZE)

# # Ensure the article_id matches between articles and image embeddings
# df_articles = df_articles.join(df_image_embeddings, on="article_id", how="left")

# def ensure_valid_embedding(embedding, target_size=1024):
#     if embedding is None:
#         return [0.0] * target_size
#     if isinstance(embedding, pl.Series):
#         embedding = embedding.to_list()
#     if not isinstance(embedding, list) or len(embedding) != target_size:
#         return [0.0] * target_size
#     return embedding

# # Replace None with zero vector explicitly before applying the function
# df_articles = df_articles.with_columns(
#     pl.when(pl.col("image_embedding").is_null())
#     .then([0.0] * 1024)
#     .otherwise(pl.col("image_embedding"))
#     .alias("image_embedding")
# )

# # Apply the function
# df_articles = df_articles.with_columns(
#     pl.col("image_embedding").apply(lambda x: ensure_valid_embedding(x, target_size=1024)).alias("image_embedding")
# )

# # Apply the function to combine and reduce embeddings
# df_articles = df_articles.with_columns(
#     pl.struct([token_col_title, "image_embedding"])
#     .apply(lambda row: combine_and_reduce_embeddings(row[token_col_title], row["image_embedding"]))
#     .alias("combined_embedding")
# )

########## IMPROVED VERSION

# # Target embedding size (same as text embedding size)
# TARGET_EMBEDDING_SIZE = 30
# INPUT_DIM = 1024

# # Load data
# df_image_embeddings = pl.read_parquet(PATH.joinpath("image_embeddings.parquet"))

# # Extract the column containing the embeddings
# image_embeddings = df_image_embeddings["image_embedding"].to_list()
# valid_embeddings = [emb for emb in image_embeddings if emb is not None and len(emb) == 1024 and not all(v == 0.0 for v in emb)]
# embedding_dataset = np.array(valid_embeddings)

# def train_linear_transform_model(embedding_dataset, input_dim=1024, target_dim=15):
#     # Input layer
#     input_layer = Input(shape=(input_dim,))
#     # Linear transformation layer
#     output_layer = Dense(target_dim, activation=None)(input_layer)

#     # Create the model
#     model = Model(inputs=input_layer, outputs=output_layer)

#     # Compile the model (no reconstruction; directly optimize for dimensionality reduction)
#     model.compile(optimizer='adam', loss='mse')

#     # Train on embeddings (X -> reduced X)
#     model.fit(embedding_dataset, np.zeros((len(embedding_dataset), target_dim)), epochs=10, batch_size=32)

#     return model

# # Ensure you replace this with the actual training dataset for image embeddings
# # e.g., embedding_dataset = np.array(df_image_embeddings["image_embedding"].to_list())
# # Train the linear transformation model only once
# linear_transform_model = train_linear_transform_model(embedding_dataset, INPUT_DIM, TARGET_EMBEDDING_SIZE//2)

# # Function to reduce embedding dimensionality using the trained linear transformation
# def reduce_embedding_with_linear_transform(embedding):
#     embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
#     reduced_embedding = linear_transform_model.predict(embedding[None, :], verbose=0)
#     return reduced_embedding[0]

# # Updated combine_embeddings function to include dimensionality reduction
# def combine_embeddings(text_embedding, image_embedding, target_size=30):
#     # Convert text_embedding to float
#     text_embedding = [float(x) for x in text_embedding]

#     # Reduce dimensionality of image embedding to half the target size
#     half_size = target_size // 2
#     image_embedding = reduce_embedding_with_linear_transform(image_embedding)

#     # Resize text embedding to match half the target size (truncate or pad)
#     if len(text_embedding) > half_size:
#         text_embedding = text_embedding[:half_size]
#     elif len(text_embedding) < half_size:
#         text_embedding = text_embedding + [0.0] * (half_size - len(text_embedding))

#     # Combine reduced text and image embeddings
#     return text_embedding + list(image_embedding)

##########

# OLDDDDDDD
# def resize_embedding(embedding, target_size):
#     if len(embedding) > target_size:
#         return embedding[:target_size]  # Truncate
#     elif len(embedding) < target_size:
#         return embedding + [0.0] * (target_size - len(embedding))  # Pad
#     return embedding

def combine_embeddings(text_embedding, image_embedding, target_size=30):
    # Convert text_embedding to float
    text_embedding = [float(x) for x in text_embedding]

    # Truncate or pad both embeddings to half the target size
    # half_size = target_size // 2
    # text_embedding = resize_embedding(text_embedding, half_size)
    # image_embedding = resize_embedding(image_embedding, half_size)

    # Combine by concatenation (results in exactly `target_size`)
    return text_embedding + image_embedding

# Ensure the article_id matches between articles and image embeddings
df_articles = df_articles.join(df_image_embeddings, on="article_id", how="left")

def ensure_valid_embedding(embedding, target_size=1024):
    if embedding is None:
        return [0.0] * target_size
    if isinstance(embedding, pl.Series):
        embedding = embedding.to_list()
    if not isinstance(embedding, list) or len(embedding) != target_size:
        return [0.0] * target_size
    return embedding

# Replace None with zero vector explicitly before applying the function
df_articles = df_articles.with_columns(
    pl.when(pl.col("image_embedding").is_null())
    .then([0.0] * 1024)
    .otherwise(pl.col("image_embedding"))
    .alias("image_embedding")
)

# Apply the function
df_articles = df_articles.with_columns(
    pl.col("image_embedding").apply(lambda x: ensure_valid_embedding(x, target_size=1024)).alias("image_embedding")
)

# Combine text and image embeddings
df_articles = df_articles.with_columns(
    pl.struct([token_col_title, "image_embedding"])
    .apply(lambda row: combine_embeddings(row[token_col_title], row["image_embedding"]))
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
    transformer_tokenizer,
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

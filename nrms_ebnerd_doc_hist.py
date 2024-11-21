from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os

from ebrec.utils._constants import *

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
    chunk_dataframe,
    split_df,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms, hparams_nrms_docvec
from ebrec.models.newsrec.nrms_docvec import NRMSModel_docvec
from ebrec.models.newsrec import NRMSModel

from utils import ebnerd_from_path, PATH, COLUMNS, DUMP_DIR, down_sample_on_users

# conda activate ./venv/; python nrms_ebnerd_doc_hist.py.py
# conda activate ./venv/; tensorboard --logdir=ebnerd_predictions/runs

model_func = NRMSModel_docvec
DT_NOW = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
SEED = 123

MODEL_NAME = f"{model_func.__name__}-{DT_NOW}"
MODEL_NAME = f"NRMSModel_docvec-2024-11-18 05-03-30"


MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
TEST_DF_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)


DATASPLIT = "ebnerd_small"
BS_TRAIN = 32
BS_TEST = 32
BATCH_SIZE_TEST_WO_B = 32
BATCH_SIZE_TEST_W_B = 4
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0

# MAX_TITLE_LENGTH = 30
EPOCHS = 5
HISTORY_SIZE = 20
NPRATIO = 4

TRAIN_FRACTION = 1.0
FRACTION_TEST = 1.0
WITH_REPLACEMENT = True
MIN_N_INVIEWS = 0  # 0 = all
MAX_N_IMPR_USERS = 0  # 0 = all

hparams_nrms_docvec.title_size = 768
hparams_nrms_docvec.history_size = HISTORY_SIZE
# MODEL ARCHITECTURE
hparams_nrms_docvec.head_num = 16
hparams_nrms_docvec.head_dim = 16
hparams_nrms_docvec.attention_hidden_dim = 200
# MODEL OPTIMIZER:
hparams_nrms_docvec.optimizer = "adam"
hparams_nrms_docvec.loss = "cross_entropy_loss"
hparams_nrms_docvec.dropout = 0.2
hparams_nrms_docvec.learning_rate = 1e-4
hparams_nrms_docvec.newsencoder_l2_regularization = 1e-4
hparams_nrms_docvec.newsencoder_units_per_layer = [256, 256, 256]

# =====

df = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=SEED)
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
#
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)


df_articles = pl.read_parquet(
    PATH.joinpath(
        "artifacts/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"
    )
)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

# =>
# =>
# train_dataloader = NRMSDataLoaderPretransform(
#     behaviors=df_train,
#     article_dict=article_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     eval_mode=False,
#     batch_size=BS_TRAIN,
# )

# val_dataloader = NRMSDataLoaderPretransform(
#     behaviors=df_validation,
#     article_dict=article_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     eval_mode=False,
#     batch_size=BS_TEST,
# )

# # CALLBACKS
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_auc", mode="max", patience=4, restore_best_weights=True
# )
# modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=MODEL_WEIGHTS,
#     monitor="val_auc",
#     mode="max",
#     save_best_only=True,
#     save_weights_only=True,
#     verbose=1,
# )
# lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=1e-6
# )
# callbacks = [lr_scheduler, early_stopping, modelcheckpoint, tensorboard_callback]

model = model_func(
    hparams=hparams_nrms_docvec,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss="categorical_crossentropy",
    metrics=["AUC"],
)
# =>
# hist = model.model.fit(
#     train_dataloader,
#     validation_data=val_dataloader,
#     epochs=EPOCHS,
#     callbacks=callbacks,
# )

print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_weights(MODEL_WEIGHTS)

# First filter: only keep users with >FILTER_MIN_HISTORY in history-size
FILTER_MIN_HISTORY = 100
# Truncate the history
HIST_SIZE = 100

# =>
df = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"), history_size=120, padding=None
    )
    .filter(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.len() >= FILTER_MIN_HISTORY)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
)

pairs = [
    (1, 256),
    (2, 256),
    (3, 256),
    (4, 256),
    (8, 256),
    (15, 128),
]

aucs = []
hists = []
for hist_size, batch_size in pairs:
    print(f"History size: {hist_size}, Batch size: {batch_size}")

    df_ = df.pipe(
        truncate_history,
        column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        history_size=hist_size,
        padding_value=0,
        enable_warning=False,
    )

    test_dataloader = NRMSDataLoader(
        behaviors=df_,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=batch_size,
    )

    scores = model.scorer.predict(test_dataloader)

    df_pred = add_prediction_scores(df_, scores.tolist())

    metrics = MetricEvaluator(
        labels=df_pred["labels"],
        predictions=df_pred["scores"],
        metric_functions=[AucScore()],
    )
    metrics.evaluate()
    auc = metrics.evaluations["auc"]
    aucs.append(auc)
    hists.append(hist_size)
    print(f"{auc} (History size: {hist_size}, Batch size: {batch_size})")

print(MODEL_WEIGHTS)
print(hists)
print(aucs)
for h, a in zip(hists, aucs):
    print(f"{a} ({h})")

from dataclasses import dataclass, field
import tensorflow as tf
import polars as pl
import numpy as np

from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

from ebrec.utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)


@dataclass
class NewsrecDataLoader(tf.keras.utils.Sequence):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    history_size: int = 20
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataLoader(NewsrecDataLoader):
    # TODO: Update this dataloader
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        self.X = self.X.with_columns(
            self.X["norm_avg_access_time"].alias("history_time_diff"), # user history
            self.X["article_age_normalized"].alias("pred_time_diff"),  # candidate news
        )
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # Dynamically compute the batch size from batch_X
        batch_size = len(batch_X)  # Number of rows in the batch
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # Repeat his_time_diff to match his_input_title
            his_time_diff = np.repeat(
                np.array(batch_X["history_time_diff"].to_list(), dtype=float),
                repeats=repeats,
                axis=0
            )
            his_time_diff = np.expand_dims(his_time_diff, axis=-1)  # Add last dimension
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            his_time_diff = np.array(batch_X["history_time_diff"].to_list(), dtype=np.float32)
            his_time_diff = np.expand_dims(his_time_diff, axis=-1)  # Add last dimension

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, his_time_diff, pred_input_title), batch_y


@dataclass
class NRMSDataLoaderPretransform(NewsrecDataLoader):
    """
    In the __post_init__ pre-transform the entire DataFrame. This is useful for
    when data can fit in memory, as it will be much faster ones training.
    Note, it might not be as scaleable.
    """

    def __post_init__(self):
        super().__post_init__()
        self.X = self.X.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        self.X = self.X.with_columns(
            self.X["norm_avg_access_time"].alias("history_time_diff"), # user history
            self.X["article_age_normalized"].alias("pred_time_diff"),  # candidate news
        )


    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        his_time_diff:  (samples, history_size, 1)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        # Extract the batch dynamically
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Dynamically compute the batch size from batch_X
        batch_size = len(batch_X)  # Number of rows in the batch

        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            his_time_diff = np.array(batch_X["history_time_diff"].to_list(), dtype=np.float32)
            his_time_diff = np.repeat(his_time_diff[:, np.newaxis], repeats=self.history_size, axis=1)
            his_time_diff = np.repeat(his_time_diff, repeats=repeats, axis=0)  # Match the repetition of his_input_title

            his_time_diff = np.expand_dims(his_time_diff, axis=-1)  # Add last dimension

            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
            pred_time_diff = np.array(batch_X["pred_time_diff"].to_list(), dtype=float)

            B, C = pred_time_diff.shape
            pred_time_diff = np.reshape(pred_time_diff, (batch_size* C, 1, 1))
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            his_time_diff = np.array(batch_X["history_time_diff"].to_list(), dtype=np.float32)
            his_time_diff = np.repeat(his_time_diff[:, np.newaxis], repeats=self.history_size, axis=1)

            pred_time_diff = np.array(batch_X["pred_time_diff"].to_list(), dtype=np.float32)
            pred_time_diff = np.expand_dims(pred_time_diff, axis=-1)  # Add last dimension

        his_input_title = np.squeeze(his_input_title, axis=2)


        return (his_input_title, his_time_diff, pred_input_title, pred_time_diff), batch_y
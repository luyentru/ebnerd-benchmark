# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2


class NRMSModel:
    """NRMS model modified to use pre-computed LLaMA embeddings.

    This version of NRMS bypasses the news encoder and directly uses LLaMA embeddings
    in the user encoder for news article representation.

    Original paper:
    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        hparams (dict): Hyperparameters for the model
        seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        hparams: dict,
        seed: int = None,
    ):
        """Initialization steps for NRMS with LLaMA embeddings."""
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            train_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self):
        """The main function to create user encoder of NRMS using LLaMA embeddings."""
        # Input shape should match the history size and LLaMA embedding dimension
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, llama_embeddings.shape[1]), dtype="float32"
        )

        # Directly use LLaMA embeddings
        click_title_presents = his_input_title

        # Apply self-attention
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        
        # Apply attention layer
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        # Create the model
        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic using LLaMA embeddings.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        # Input for user history with LLaMA embeddings
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, llama_embeddings.shape[1]),
            dtype="float32",
        )
        
        # Input for candidate news articles with LLaMA embeddings
        pred_input_title = tf.keras.Input(
            shape=(None, llama_embeddings.shape[1]),
            dtype="float32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(1, llama_embeddings.shape[1]),
            dtype="float32",
        )
        pred_title_one_reshape = tf.keras.layers.Reshape((llama_embeddings.shape[1],))(
            pred_input_title_one
        )

        # Use the user encoder
        user_present = self._build_userencoder()(his_input_title)
        
        # Process candidate news articles
        news_present = tf.keras.layers.TimeDistributed(lambda x: x)(pred_input_title)
        news_present_one = pred_title_one_reshape

        # Compute predictions
        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        # Define models
        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer

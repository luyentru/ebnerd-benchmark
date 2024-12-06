# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2


class NRMSModel:
    def __init__(
        self,
        hparams: dict,
        seed: int = None,
    ):
        """Initialize NRMS model with LLaMA embeddings."""
        self.hparams = hparams
        self.seed = seed
        self.embedding_dim = hparams.embedding_dim  # LLaMA embedding dimension (4096)
        
        # Set random seeds
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Build and compile model
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate)
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _build_userencoder(self):
        """Creates user encoder for NRMS using projected LLaMA embeddings."""
        # Input is already projected to attention_hidden_dim
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.attention_hidden_dim),
            dtype="float32"
        )

        # Self attention with correct dimensions
        y = SelfAttention(
            head_num=self.hparams.head_num,  # typically 20
            head_dim=self.hparams.attention_hidden_dim // self.hparams.head_num,  # 200 // 20 = 10
            seed=self.seed
        )([his_input_title] * 3)
        
        # Final attention layer
        user_present = AttLayer2(
            hidden_dim=self.hparams.attention_hidden_dim,  # 200
            seed=self.seed
        )(y)

        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_nrms(self):
        """Creates NRMS model using LLaMA embeddings."""
        # Inputs are LLaMA embeddings (4096 dim)
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.embedding_dim),  # (20, 4096)
            dtype="float32",
        )
        pred_input_title = tf.keras.Input(
            shape=(None, self.embedding_dim),  # (None, 4096)
            dtype="float32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(1, self.embedding_dim),  # (1, 4096)
            dtype="float32",
        )
        
        # Project all inputs from LLaMA dim (4096) to attention_hidden_dim (200)
        projection_layer = tf.keras.layers.Dense(
            self.hparams.attention_hidden_dim,  # 200
            use_bias=False,
            name="llama_projection"
        )
        
        # Project all inputs
        his_projected = tf.keras.layers.TimeDistributed(projection_layer)(his_input_title)  # (20, 200)
        pred_projected = tf.keras.layers.TimeDistributed(projection_layer)(pred_input_title)  # (None, 200)
        pred_one_projected = projection_layer(pred_input_title_one)  # (1, 200)
        
        # Get user representation
        user_present = self._build_userencoder()(his_projected)  # (200,)
        
        # Process candidate news
        news_present = pred_projected  # (None, 200)
        news_present_one = tf.squeeze(pred_one_projected, axis=1)  # (200,)

        # Compute predictions
        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])  # (None,)
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([
            tf.expand_dims(news_present_one, axis=0),  # (1, 200)
            tf.expand_dims(user_present, axis=0)  # (1, 200)
        ])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        # Define models
        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer

    def _build_graph(self):
        """Build the complete model graph."""
        model, scorer = self._build_nrms()
        return model, scorer

    def _get_loss(self, loss):
        """Get the loss function."""
        if loss == "cross_entropy_loss":
            return "categorical_crossentropy"
        elif loss == "binary_cross_entropy":
            return "binary_crossentropy"
        else:
            raise ValueError(f"Unknown loss type: {loss}")

    def _get_opt(self, optimizer="adam", lr=0.001):
        """Get the optimizer."""
        if optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer}")
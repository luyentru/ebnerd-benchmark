# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, embed_dim, seed=42):
        super(Time2Vec, self).__init__()
        self.embed_dim = embed_dim
        self.seed = seed

    def build(self, input_shape):
        # Use initializers with seed for reproducibility
        initializer = tf.keras.initializers.RandomUniform(seed=self.seed)

        # Define trainable weights
        self.wb = self.add_weight(name='wb', shape=(1,), initializer=initializer, trainable=True)
        self.bb = self.add_weight(name='bb', shape=(1,), initializer=initializer, trainable=True)
        self.wa = self.add_weight(name='wa', shape=(self.embed_dim - 1,), initializer=initializer, trainable=True)
        self.ba = self.add_weight(name='ba', shape=(self.embed_dim - 1,), initializer=initializer, trainable=True)

    def call(self, inputs):
        # Ensure inputs have shape [batch_size, time_embed_dim - 1]
        inputs = tf.expand_dims(inputs, axis=-1)  # Shape: [batch_size, 1]
        
        # Linear component
        time_linear = self.wb * inputs + self.bb  # Shape: [batch_size, 1]

        # Periodic component
        time_periodic = tf.math.sin(inputs * self.wa + self.ba)  # Element-wise operation

        # Combine linear and periodic components
        return tf.concat([time_linear, time_periodic], axis=-1)  # Shape: [batch_size, time_embed_dim]


class NRMSModel:
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
    """

    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            # Xavier Initialization
            initializer = GlorotUniform(seed=self.seed)
            self.word2vec_embedding = initializer(shape=(vocab_size, word_emb_dim))
            # self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

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

    def _build_userencoder(self, titleencoder: tf.keras.Model) -> tf.keras.Model:
        """
        Build the user encoder for NRMS (Neural News Recommendation with Multi-Head Self-Attention).

        The user encoder processes the user's interaction history to generate a user representation.
        It combines the article embeddings from the user's history with temporal embeddings derived
        from the recency of interactions (time differences).

        Args:
            titleencoder (tf.keras.Model): The news encoder model that generates article embeddings.

        Returns:
            tf.keras.Model: A Keras model that encodes the user's interaction history into a latent representation.
        """
        # Input: User's historical article clicks (batch_size, history_size, title_size)
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )

        # Input: Time differences for the user's historical interactions (batch_size, history_size, 1)
        time_diff_input = tf.keras.Input(
            shape=(self.hparams.history_size, 1), dtype="float32"
        )

        # Embed titles in the user's history using the title encoder
        # Output shape: (batch_size, history_size, embedding_dim)
        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )

        # Embed time differences using Time2Vec to encode both linear and periodic components
        # Output shape: (batch_size, history_size, time_embed_dim)
        time2vec_layer = Time2Vec(embed_dim=50, seed=self.seed)
        time_embeddings = tf.keras.layers.TimeDistributed(time2vec_layer)(time_diff_input)

        # Remove extra dimension added by Time2Vec
        # Final shape: (batch_size, history_size, time_embed_dim)
        time_embeddings = tf.squeeze(time_embeddings, axis=-2)

        # Apply self-attention to the time embeddings (e.g., to weigh importance of time values)
        # Shape remains: (batch_size, history_size, time_embed_dim)
        time_attention = tf.keras.layers.Attention()([time_embeddings, time_embeddings])

        # Concatenate article embeddings with time embeddings along the last axis
        # Combined shape: (batch_size, history_size, embedding_dim + time_embed_dim)
        combined_embeddings = tf.concat([click_title_presents, time_attention], axis=-1)

        # Apply multi-head self-attention on the combined embeddings
        # Output shape: (batch_size, history_size, attention_hidden_dim)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [combined_embeddings] * 3
        )

        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        # Define the user encoder model
        model = tf.keras.Model([his_input_title, time_diff_input], user_present, name="user_encoder")
        return model


    def _build_newsencoder(self) -> tf.keras.Model:
        """
        Build the news encoder for NRMS.

        The news encoder processes article content (e.g., titles) into a latent representation using
        an embedding layer, self-attention, and dense layers.

        Returns:
            tf.keras.Model: A Keras model that encodes article content into a latent representation.
        """
        # Embedding layer to encode words into dense vectors
        embedding_layer = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0],  # Vocabulary size
            self.word2vec_embedding.shape[1],  # Embedding dimension
            weights=[self.word2vec_embedding],  # Pre-trained embeddings
            trainable=True,  # Allow fine-tuning
        )

        # Input: Article titles represented as sequences of word indices (batch_size, title_size)
        sequences_input_title = tf.keras.Input(shape=(self.hparams.title_size,), dtype="int32")

        # Embed the title using the embedding layer
        # Output shape: (batch_size, title_size, embedding_dim)
        embedded_sequences_title = embedding_layer(sequences_input_title)

        # Apply dropout for regularization
        y = tf.keras.layers.Dropout(self.hparams.dropout)(embedded_sequences_title)

        # Apply multi-head self-attention to learn contextual word representations
        # Output shape: (batch_size, title_size, attention_dim)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [y, y, y]
        )

        # Apply several dense layers for further transformation
        for layer in [400, 400, 400]:
            y = tf.keras.layers.Dense(units=layer, activation="relu", kernel_regularizer=l2(1e-3))(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Dropout(self.hparams.dropout)(y)

        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        # Define the news encoder model
        model = tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model


    def _build_nrms(self) -> tuple[tf.keras.Model, tf.keras.Model]:
    # Inputs for user history and candidate articles
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        pred_input_title = tf.keras.Input(
            shape=(None, self.hparams.title_size), dtype="int32"  # For multiple candidates
        )
        time_diff_input = tf.keras.Input(
            shape=(self.hparams.history_size, 1), dtype="float32"  # Time differences for user history
        )
        pred_time_diff_input = tf.keras.Input(
            shape=(None, 1), dtype="float32"  # Time differences for candidate news
        )
        pred_input_title_one = tf.keras.Input(
            shape=(1, self.hparams.title_size), dtype="int32"  # For a single candidate
        )
        pred_time_diff_input_one = tf.keras.Input(
            shape=(1, 1), dtype="float32"  # Time difference for a single candidate
        )

        # Flatten the single candidate input
        pred_title_one_reshape = tf.keras.layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )

        # Build the news and user encoders
        titleencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        # Generate user representations
        user_present = self.userencoder([his_input_title, time_diff_input])

        # Generate candidate news representations
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        # Generate candidate time embeddings
        time2vec_layer = Time2Vec(embed_dim=50, seed=self.seed)
        pred_time_embeddings = tf.keras.layers.TimeDistributed(time2vec_layer)(pred_time_diff_input)
        pred_time_embeddings = tf.squeeze(pred_time_embeddings, axis=-2)

        pred_time_embedding_one = time2vec_layer(pred_time_diff_input_one)
        pred_time_embedding_one = tf.squeeze(pred_time_embedding_one, axis=[1, 2])  # Remove dimensions 1 and 2

        # Combine candidate news representations with time embeddings
        combined_news_present = tf.concat([news_present, pred_time_embeddings], axis=-1)
        combined_news_present_one = tf.concat([news_present_one, pred_time_embedding_one], axis=-1)

        projected_news_present = tf.keras.layers.Dense(
        #units=self.hparams.attention_hidden_dim,  # Match dimension of user_present (400)
        units=400,
        activation="relu"
        )(combined_news_present)
        
        projected_news_present_one = tf.keras.layers.Dense(
        #units=self.hparams.attention_hidden_dim,  # Match dimension of user_present (400)
        units=400,
        activation="relu"
        )(combined_news_present_one)

        # Compute relevance scores
        preds = tf.keras.layers.Dot(axes=-1)([projected_news_present, user_present])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([projected_news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        # Define the NRMS model and scorer model
        model = tf.keras.Model(
            [his_input_title, time_diff_input, pred_input_title, pred_time_diff_input],
            preds
        )
        scorer = tf.keras.Model(
            [his_input_title, time_diff_input, pred_input_title_one, pred_time_diff_input_one],
            pred_one
        )

        return model, scorer
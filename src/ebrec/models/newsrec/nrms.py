import tensorflow as tf  # TensorFlow for building models and layers
import numpy as np  # NumPy for handling numerical operations and random seeds

# TensorFlow Keras imports for model building
from tensorflow.keras.layers import Input, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import Callback

# Custom imports for NRMS-specific layers
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention


class NRMSModel:
    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        # Validate hyperparameters
        required_hparams = [
            'history_size', 'embedding_dim', 'head_num', 'head_dim',
            'attention_hidden_dim', 'loss', 'optimizer', 'learning_rate'
        ]
        for param in required_hparams:
            if param not in hparams:
                raise ValueError(f"Missing required hyperparameter: {param}")

        self.hparams = hparams
        self.seed = seed

        # Set random seed for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Initialize user encoder
        self.userencoder = self._build_userencoder()

        # Build and compile model
        self.model, self.scorer = self._build_nrms()
        data_loss = self._get_loss(self.hparams['loss'])
        train_optimizer = self._get_opt(
            optimizer=self.hparams['optimizer'], lr=self.hparams['learning_rate']
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
        """Get optimizer instance."""
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            return optimizer
        elif optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

    def _build_userencoder(self):
        """Create user encoder using precomputed embeddings."""
        click_title_presents = tf.keras.Input(
            shape=(self.hparams['history_size'], self.hparams['embedding_dim']), dtype="float32"
        )

        y = SelfAttention(self.hparams['head_num'], self.hparams['head_dim'], seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(self.hparams['attention_hidden_dim'], seed=self.seed)(y)

        model = tf.keras.Model(click_title_presents, user_present, name="user_encoder")
        return model

    def _build_nrms(self):
        """Build NRMS's logic."""
        his_input_title = tf.keras.Input(
            shape=(self.hparams['history_size'], self.hparams['embedding_dim']), dtype="float32"
        )  # Precomputed embeddings

        pred_input_title = tf.keras.Input(
            shape=(None, self.hparams['embedding_dim']), dtype="float32"
        )  # Precomputed candidate embeddings

        pred_input_title_one = tf.keras.Input(
            shape=(self.hparams['embedding_dim'],), dtype="float32"
        )

        # Generate user representation
        user_present = self.userencoder(his_input_title)

        # Add Dense layers to align dimensions
        user_present_projected = tf.keras.layers.Dense(self.hparams['embedding_dim'])(user_present)
        pred_input_title_projected = tf.keras.layers.Dense(self.hparams['embedding_dim'])(pred_input_title)
        pred_input_title_one_projected = tf.keras.layers.Dense(self.hparams['embedding_dim'])(pred_input_title_one)

        # Compute dot product for multiple predictions
        preds = tf.keras.layers.Dot(axes=-1)([pred_input_title_projected, user_present_projected])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        # Compute dot product for single prediction
        pred_one = tf.keras.layers.Dot(axes=-1)([pred_input_title_one_projected, user_present_projected])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        # Define models
        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer


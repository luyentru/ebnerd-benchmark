#
DEFAULT_TITLE_SIZE = 30
DEFAULT_BODY_SIZE = 40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768


def print_hparams(hparams_class):
    for attr, value in hparams_class.__annotations__.items():
        # Print attribute names and values
        print(f"{attr}: {getattr(hparams_class, attr)}")


def hparams_to_dict(hparams_class) -> dict:
    params = {}
    for attr, value in hparams_class.__annotations__.items():
        params[attr] = getattr(hparams_class, attr)
    return params


class hparams_nrms:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 20
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 1e-4


class hparams_nrms_docvec:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_DOCUMENT_SIZE
    history_size: int = 20
    # MODEL ARCHITECTURE
    head_num: int = 16
    head_dim: int = 16
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 1e-4
    newsencoder_units_per_layer: list[int] = [512, 512, 512]
    newsencoder_l2_regularization: float = 1e-4

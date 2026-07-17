import random
import numpy as np
import tensorflow as tf
from src.config import SEED

def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.config.experimental.enable_op_determinism()
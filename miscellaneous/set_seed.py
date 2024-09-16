import os
import tensorflow as tf
from numpy.random import seed
import random

def set_seed(seed_num) -> None:
    random.seed(seed_num)
    seed(seed_num)
    tf.random.set_seed(seed_num)
    tf.experimental.numpy.random.seed(seed_num)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_num)
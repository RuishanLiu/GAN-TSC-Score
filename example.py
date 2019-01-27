import numpy as np
import tensorflow as tf
from compression_score import get_compression_score

# Load data for Compression Score evaluation. Data in the shape of (50000, 32, 32, 3)
(data, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Calculate Compression Score
score, score_var = get_compression_score(data, path_teacher='compression_teacher.npy', n_iter=3,  retrain_real=False, gpu=0)
print('Compression score %.3f +/- %.3f' % (score, score_var))
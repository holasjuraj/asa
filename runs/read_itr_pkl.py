import os
import tensorflow as tf
import joblib

# If GPUs are blocked by another user, force use specific GPU (0 or 1),
# or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path = 'data/local/asa-test/instant-run/itr_11.pkl'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config).as_default():
    data = joblib.load(path)

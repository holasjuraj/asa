import os
import tensorflow as tf
import joblib

# Snippet for starting interactive TF session and reading exported training snapshot

# If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

path = 'data/local/asa-test/itr_11.pkl'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
saved_data = joblib.load(path)


sess.close()

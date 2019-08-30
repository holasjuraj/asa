import tensorflow as tf
import joblib

path = 'data/local/asa-test/instant-run/itr_11.pkl'
with tf.Session().as_default():
    data = joblib.load(path)

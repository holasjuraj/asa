import os
import tensorflow as tf
import dill
import joblib
from garage.misc.tensor_utils import unflatten_tensors

# Snippet for starting interactive TF session and reading exported training snapshot

# If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

path = '/home/h/holas3/garage/data/local/asa-test/instant_run/itr_0.pkl'
path = '/home/h/holas3/garage/data/archive/asa-test/2019_10_18-12_35--Basic_run_25itrs_maps12_b5000--s1/itr_3.pkl'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
# saved_data = joblib.load(path)
with open(path, 'rb') as file:
    saved_data = dill.load(file)

# Read weights
old_top_policy = saved_data['policy']
otp_weight_values = unflatten_tensors(
    old_top_policy.get_param_values(),
    old_top_policy.get_param_shapes()
)
otp_weights = list(zip(
    [p.name for p in old_top_policy.get_params()],
    otp_weight_values
))

for name, value in otp_weights:
    print(name, value.shape)
    print(value.T)


sess.close()

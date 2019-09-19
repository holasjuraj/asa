import tensorflow as tf

from garage.core import Serializable
from garage.tf.core import layers as ly
from garage.tf.core import LayersPowered

from garage.tf.core.network import MLP as GarageMLP


class MLP(GarageMLP):
    def __init__(
            self,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            output_nonlinearity,
            hidden_w_tf_vars,
            hidden_b_tf_vars,
            output_w_tf_var,
            output_b_tf_var,
            name=None,
            input_var=None,
            input_layer=None,
            input_shape=None,
            batch_normalization=False,
            weight_normalization=False,
    ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name, "MLP"):
            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, ) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = ly.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = ly.DenseLayer(
                    l_hid,
                    num_units=hidden_size,    # TODO Get information from hidden_w_tf_vars[idx]
                                              #      and get rid of hidden_sizes parameter
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    w=hidden_w_tf_vars[idx],  # TODO Here we can provide
                    b=hidden_b_tf_vars[idx],  #      tf.Tensor or tf.Variable
                    weight_normalization=weight_normalization)
                if batch_normalization:
                    l_hid = ly.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out = ly.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                w=output_w_tf_var,  # TODO Here we can provide
                b=output_b_tf_var,  #      tf.Tensor or tf.Variable
                weight_normalization=weight_normalization)
            if batch_normalization:
                l_out = ly.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out

            LayersPowered.__init__(self, l_out)

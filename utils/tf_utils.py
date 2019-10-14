import tensorflow as tf


def uninitialized_variables_initializer(sess=None, variables=None):
    """
    Op to initialize only uninitialized TF variables
    :param sess: TF session
    :param variables: list of TF variables from which to choose the uninitialized ones
    :return: TF op
    """
    if sess is None:
        sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    is_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in variables])
    not_initialized_vars = [var for (var, inited) in
                            zip(variables, is_initialized) if not inited]
    return tf.variables_initializer(not_initialized_vars)

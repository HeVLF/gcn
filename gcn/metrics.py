import tensorflow as tf
#tf.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=tf.stop_gradient(labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    tf.compat.v1.summary.histogram('preds', preds)
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
    return tf.reduce_mean(input_tensor=accuracy_all)

def masked_auc(preds, labels, mask):

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    
    # Create an instance of AUC metric
    auc_metric = tf.keras.metrics.AUC()

    # Update the metric using your predictions and ground truth labels
    auc_metric.update_state(labels, preds)

    # Get the computed AUC value
    auc = auc_metric.result()

    #auc = tf.contrib.metrics.streaming_auc(preds, labels) #tensorflow v1

    return tf.reduce_mean(input_tensor=auc)


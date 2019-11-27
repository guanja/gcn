import tensorflow as tf
from tensorflow.metrics import recall, precision


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


# guanja
def masked_recall(preds, labels, mask):
    """Recall with masking.

    Recall is defined as: TP/(TP+FN)
    By multiplying with mask, we are only increasing the count of TN, which
    does not count towards the REC, hence we can
    1. mask
    2. use tf.metrics.recall(labels, predictions)

    Args:
        preds: predictions, size=(n_samples, n_classes)
        labels: true labels, size=(n_samples, n_classes)
        mask: binary mask, size=(n_samples), indicates which samples should be 
              used to evaluate the recall.
    
    Returns:
        (recall, op_recall)
    """
    predictions = tf.argmax(preds, 1)
    labels = tf.argmax(labels, 1)
    # cast to float32.
    predictions = tf.cast(predictions, tf.float32)
    labels = tf.cast(labels, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # mask.
    predictions *= mask
    labels *= mask
    return recall(labels, predictions)


# guanja
def masked_precision(preds, labels, mask):
    """Precision with masking.

    Precision is defined as: TP/(TP+FP)
    By multiplying with mask, we are only increasing the count of TN, which
    does not count towards the PREC, hence we can
    1. mask
    2. use tf.metrics.precision(labels, predictions)

    Args:
        preds: predictions, size=(n_samples, n_classes)
        labels: true labels, size=(n_samples, n_classes)
        mask: binary mask, size=(n_samples), indicates which samples should be 
              used to evaluate the recall.
    
    Returns:
        (precision, op_precision)
    """
    predictions = tf.argmax(preds, 1)
    labels = tf.argmax(labels, 1)
    # cast to float32.
    predictions = tf.cast(predictions, tf.float32)
    labels = tf.cast(labels, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # mask.
    predictions *= mask
    labels *= mask
    return precision(labels, predictions)

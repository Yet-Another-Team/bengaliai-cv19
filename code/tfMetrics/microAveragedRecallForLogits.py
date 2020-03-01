import tensorflow as tf

class RecallForLogits(tf.keras.metrics.Recall):
    def __init__(self, name='recall', **kwargs):
        super(RecallForLogits, self).__init__(name=name, **kwargs)            

    def update_state(self, y_true, y_pred_logit, sample_weight=None):
        probs = tf.nn.softmax(y_pred_logit)
        print("probs shape is {0}".format(probs.shape))
        S = tf.shape(y_true)
        idx_max = tf.math.argmax(probs,axis=-1)
        y_pred = tf.one_hot(idx_max,S[-1],1.0,0.0)
        print("y_pred shape is {0}".format(y_pred.shape))
        super().update_state(y_true, y_pred, sample_weight)            

    def result(self):
        return super().result()


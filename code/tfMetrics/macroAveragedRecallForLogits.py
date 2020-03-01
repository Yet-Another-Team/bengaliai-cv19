import tensorflow as tf
import numpy as np

class RecallForLogits(tf.keras.metrics.Metric):
    def __init__(self, maxClassesCount=168, name='recall', **kwargs):
        self.M = maxClassesCount
        #print("Tracking macro-averaged recall for {0} classes".format(self.M))
        super(RecallForLogits, self).__init__(name=name, **kwargs)            
        self.true_positives = self.add_weight(name='tp', initializer='zeros',shape=(self.M,))
        self.false_negatives = self.add_weight(name='fn', initializer='zeros',shape=(self.M,))
    
    def reset_states(self):
        """Resets all of the metric state variables.
        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        # value should be a Numpy array
        zeros = np.zeros(self.M)
        tf.keras.backend.batch_set_value([(v, zeros) for v in self.variables])


    def update_state(self, y_true, y_pred_logit, sample_weight=None):
        # shape is: B x M(ClassesCount)
        #y_pred_shape = tf.shape(y_pred_logit)
        #B = y_pred_shape[0]
        idx_max = tf.math.argmax(y_pred_logit,1) # (B,)
        
        y_pred_bool = tf.one_hot(idx_max,self.M,on_value=True, off_value=False) # BxM

        y_true_bool = tf.cast(y_true,dtype=tf.bool)
        
        not_pred_bool = tf.math.logical_not(y_pred_bool)


        localTP = tf.math.reduce_sum(tf.cast(tf.math.logical_and(y_pred_bool,y_true_bool),dtype=tf.float32),0) # along Batch
        localFN = tf.math.reduce_sum(tf.cast(tf.math.logical_and(not_pred_bool,y_true_bool),dtype=tf.float32),0) # along Batch

        # print("true_positives shape: {0}".format(self.true_positives.shape))
        # print("false_negatives shape: {0}".format(self.false_negatives.shape))
        # print("localTP shape: {0}".format(localTP.shape))
        # print("localFN shape: {0}".format(localFN.shape))

        self.true_positives.assign_add(localTP)
        self.false_negatives.assign_add(localFN)   

    def result(self):
        #print("result self.true_positives shape: {0}".format(self.true_positives.shape))
        nom = tf.cast(self.true_positives,dtype=tf.float32) # shape (M,)
        denom = tf.cast(self.true_positives + self.false_negatives,dtype=tf.float32) # shape (M,)  
        perClassRecall = nom/denom # NaN can emerge  
        nansIndicator = tf.math.is_nan(perClassRecall)
        perClassRecallNoNAN = tf.where(nansIndicator,tf.zeros_like(perClassRecall), perClassRecall)
        recallSum = tf.reduce_sum(perClassRecallNoNAN)
        nonNanCount = tf.reduce_sum(tf.cast(tf.logical_not(nansIndicator),tf.float32))
        return recallSum/nonNanCount
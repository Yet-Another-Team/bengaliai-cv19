import tensorflow as tf

import sys
import os
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet
from tfDataIngest import tfDataSetParquetAnnotateTrain as tfDsParquetAnnotation
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from models.MobileNetV2 import GetModel

inputDataDir = sys.argv[1]
validationFile = sys.argv[2]
experiment_output_dir = sys.argv[3]
dropoutRate = 0.2
batchSize = 8
seed = 313143

print("validation set samples listing: {0}".format(validationFile))

valDf = pd.read_csv(validationFile)
valIds = set(valDf.image_id)
print("{0} samples will be used for validation".format(len(valIds)))

if __name__ == "__main__":    
    tf.random.set_seed(seed+563)

    print("Data dir is {0}".format(inputDataDir))
    dataFileNames = glob("{0}/train*.parquet".format(inputDataDir))
    trainLabelsFileName = "{0}/train.csv".format(inputDataDir)

    N = len(pd.read_csv(trainLabelsFileName))
    #N = 5000
    print("There are {0} training samples in total".format(N))

    print("Parquet files count is {0}".format(len(dataFileNames)))
    print("First is {0}".format(dataFileNames[0]))

    def constructAllSamplesDs():
        ds = tfDsParquet.create_parquet_dataset(dataFileNames)
        ds = tfDsParquetAnnotation.annotate(ds,trainLabelsFileName)  
        return ds  


    # reshaping to match the input shape
    def prepareInput(_,labels,pixels):
        #pixels = tf.cast(pixels, tf.float32)
        root,vowel,consonant = tf.unstack(labels,3)
        root = tf.one_hot(root, 168, dtype=tf.uint8)
        vowel = tf.one_hot(vowel, 11, dtype=tf.uint8)
        consonant = tf.one_hot(consonant, 7, dtype=tf.uint8)

        colored = tf.tile(tf.expand_dims(pixels,-1),[1,1,3])

        pixels = tf.image.resize(colored, [224,224], method='gaussian')
        #HEIGHT = 137
        #WIDTH = 236

        #pixels = tf.pad(colored,[[43,44],[0,0],[0,0]])[:,6:230,:]
        labelsDict = {
            "root": tf.reshape(root,(168,)),
            "vowel": tf.reshape(vowel,(11,)),
            "consonant": tf.reshape(consonant,(7,))
        }
        return pixels, labelsDict    

    def inValidationFilter(ident):            
        identBytes = ident.numpy()
        identStr = identBytes.decode('utf-8')
        return identStr in valIds
    def inValFilter(ident,_dummy_1,_dummy_2):
        return tf.py_function(inValidationFilter, [ident], (tf.bool))
    def inTrainFilter(ident,_dummy_1,_dummy_2):
        return not(tf.py_function(inValidationFilter, [ident], (tf.bool)))
    
    allDs = constructAllSamplesDs()
    allDs = allDs.take(N)
    allDs = allDs.cache()

    trDs = allDs.filter(inTrainFilter)    
    trDs = trDs.map(prepareInput) #tf.data.experimental.AUTOTUNE
    #trDs = trDs.take(1000)
    #trDs = trDs.cache(os.path.join(cacheLocation,'trCache'))
    

    print("Caching all DS")
    #for element in tqdm(allDs.as_numpy_iterator(),ascii=True,total=N):
    for element in allDs.as_numpy_iterator():
       ()

    trDs = trDs.repeat()
    #trDs = trDs.prefetch(128)
    trDs = trDs.shuffle(512,seed=seed+123678, reshuffle_each_iteration=True)    
    trDs = trDs.batch(batchSize)
    #trDs = trDs.prefetch(128)

    #valDs = constructAllSamplesDs()
    valDs = allDs.filter(inValFilter)
    valDs = valDs.map(prepareInput)    
    valDs = valDs.batch(batchSize)
    #valDs = valDs.cache(os.path.join(cacheLocation,'vaCache'))
    #valDs = valDs.cache()

    print("Training dataSet is {0}".format(trDs))
    print("Validation dataSet is {0}".format(valDs))

    model,cnn = GetModel(dropoutRate, seed+44)

    print("Model constructed")
    print(model.summary())
    
    def catCeFromLogitsDoubled(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)*2.0
    def catCeFromLogits(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)    

    class RecallForLogits2(tf.keras.metrics.Metric):
        def __init__(self, name='recall', **kwargs):
            self.M = 200
            super(RecallForLogits, self).__init__(name=name, **kwargs)            
            self.true_positives = self.add_weight(name='tp', initializer='zeros',shape=(self.M,))
            self.false_negatives = self.add_weight(name='fn', initializer='zeros',shape=(self.M,))

        def update_state(self, y_true, y_pred_logit, sample_weight=None):
            # shape is: B x M(ClassesCount)
            #y_pred_shape = tf.shape(y_pred_logit)
            #B = y_pred_shape[0]
            idx_max = tf.math.argmax(y_pred_logit,1) # (B,)
            
            y_pred_bool = tf.one_hot(idx_max,self.M,on_value=True, off_value=False) # BxM

            #print("y_pred_bool shape: {0}".format(y_pred_bool.shape))
            #y_pred_bool = tf.expand_dims(y_pred_bool,0)
            #y_pred_bool = tf.tile(y_pred_bool,[B,1])

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
            print("result self.true_positives shape: {0}".format(self.true_positives.shape))
            nom = tf.cast(self.true_positives,dtype=tf.float32) # shape (M,)
            denom = tf.cast(self.true_positives + self.false_negatives,dtype=tf.float32) # shape (M,)            
            print("denom shape: {0}".format(denom.shape))
            perClassRecall = tf.cond(denom < 0.5, lambda: tf.zeros([self.M],dtype=tf.float32), lambda: nom/denom)
            print("perClassRecall shape: {0}".format(perClassRecall.shape))
            macroRecallNom = tf.math.reduce_sum(perClassRecall)
            print("macroRecallNom shape: {0}".format(macroRecallNom.shape))
            macroRecallDenom = tf.reduce_sum(tf.cast(denom > 0.0,dtype=tf.float32))
            print("macroRecallDenom shape: {0}".format(macroRecallDenom.shape))
            macroRecall = macroRecallNom/macroRecallDenom
            print("macroRecall shape: {0}".format(macroRecall.shape))
            return macroRecall
    
    class RecallForLogits(tf.keras.metrics.Recall):
        def __init__(self, name='recall', **kwargs):
            super(RecallForLogits, self).__init__(name=name, **kwargs)            

        def update_state(self, y_true, y_pred, sample_weight=None):
            probs = tf.nn.softmax(y_pred)
            super().update_state(y_true, probs, sample_weight)            

        def result(self):
            return super().result()

    model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          #optimizer=tf.keras.optimizers.RMSprop(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss= {
              "root":catCeFromLogitsDoubled,
              "vowel":catCeFromLogits,
              "consonant":catCeFromLogits
          },
          metrics=[RecallForLogits()]
          )
    print("model compiled")

    print(model.summary())

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(experiment_output_dir,'training_log.csv'),append=False)
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=int(5), monitor='root_loss',mode='min'),
        # Write TensorBoard logs to `./logs` directory
        # tf.keras.callbacks.TensorBoard(log_dir=experiment_output_dir, histogram_freq = 0, profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(experiment_output_dir,"weights.hdf5"),
            save_best_only=True,
            verbose=True,
            mode='min',
            save_weights_only=True,
            monitor='root_loss'),
        tf.keras.callbacks.TerminateOnNaN(),
        csv_logger,
        #reduce_lr
    ]

    spe = (N-len(valIds))//batchSize
    #spe = N//batchSize
    print("Steps per epoch {0}".format(spe))
    fitHisotry = model.fit(x = trDs, \
      validation_data = valDs,      
      verbose = 2,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= spe,
      #steps_per_epoch= N//batchSize,
      #steps_per_epoch= 4096,
      #epochs=int(10000)
      epochs = 10
      )    
    print("Done")
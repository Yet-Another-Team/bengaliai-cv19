import tensorflow as tf
import tfDataIngest.tfDataSetParquet as tfDsParquet
import tfDataIngest.tfDataSetParquetAnnotateTrain as tfDsParquetAnnotation
import os
import sys
import pandas as pd
from glob import glob
from models.DenseNet121 import GetModel

inputDataDir = sys.argv[1]
validationFile = sys.argv[2]
experiment_output_dir = sys.argv[3]
dropoutRate = 0.2
batchSize = 8

# valDf = pd.read_csv(validationFile)
# valIds = set(valDf.image_id)
# print("{0} samples will be used for validation".format(len(valIds)))

if __name__ == "__main__":    
    print("Data dir is {0}".format(inputDataDir))
    dataFileNames = glob("{0}/train*.parquet".format(inputDataDir))
    trainLabelsFileName = "{0}/train.csv".format(inputDataDir)

    N = len(pd.read_csv(trainLabelsFileName))
    print("There are {0} training samples in total".format(N))

    print("Parquet files {0}".format(len(dataFileNames)))
    print("First is {0}".format(dataFileNames[0]))
    ds = tfDsParquet.create_parquet_dataset(dataFileNames)
    ds = tfDsParquetAnnotation.annotate(ds,trainLabelsFileName)    

    # reshaping to match the input shape
    def prepareInput(ident,labels,pixels):
        pixels = tf.cast(pixels, tf.float16)
        root,vowel,consonant = tf.unstack(labels,3)
        root = tf.one_hot(root, 168, dtype=tf.float16)
        vowel = tf.one_hot(vowel, 11, dtype=tf.float16)
        consonant = tf.one_hot(consonant, 7, dtype=tf.float16)

        colored = tf.tile(tf.expand_dims(pixels,-1),[1,1,3])

        pixels = tf.image.resize(colored, [224,224], method='gaussian')
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
    
    #trDs = ds.filter(inTrainFilter)
    trDs = ds
    trDs = trDs.map(prepareInput, num_parallel_calls=1) #tf.data.experimental.AUTOTUNE
    #trDs = trDs.shuffle(8192,seed=123678, reshuffle_each_iteration=True)
    trDs = trDs.batch(batchSize)
    #trDs = trDs.prefetch(8)
    trDs = trDs.repeat()

    # valDs = ds.filter(inValFilter)
    # valDs = valDs.map(prepareInput, num_parallel_calls=1)    
    # valDs = valDs.batch(batchSize)

    print("DataSet is {0}".format(ds))

    model = GetModel(dropoutRate)

    print("Model constructed")
    print(model.summary())
    
    def rootLoss(y_true, y_pred):
        #print("root loss {0} {1}".format(y_true,y_pred))
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)*2.0
    def vowelLoss(y_true, y_pred):
        #print("vowelLoss loss {0} {1}".format(y_true,y_pred))
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    def consonantLoss(y_true, y_pred):
        #print("consonant loss {0} {1}".format(y_true,y_pred))
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

    def recall_for_logits(y_true, y_pred, sample_weight=None):
        probs = tf.sigmoid(y_pred)
        return tf.keras.metrics.Recall(y_true, probs, sample_weight, thresholds=0.5)              

    model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          #optimizer=tf.keras.optimizers.RMSprop(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss= {
              "root":rootLoss,
              "vowel":vowelLoss,
              "consonant":consonantLoss
          },
          metrics={
              "root":recall_for_logits,
              "vowel":recall_for_logits,
              "consonant":recall_for_logits
            }
          )
    print("model compiled")

    print(model.summary())

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(experiment_output_dir,'training_log.csv'),append=True)
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        #tf.keras.callbacks.EarlyStopping(patience=int(7), monitor='loss',mode='min'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=experiment_output_dir, histogram_freq = 0, profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment_output_dir,"weights.{epoch:02d}-{loss:.5f}.hdf5"),verbose=True, save_weights_only=True),
        tf.keras.callbacks.TerminateOnNaN(),
        csv_logger,
        #reduce_lr
    ]

    #print("Fitting")
    model.fit(x = trDs, \
      #validation_data = valDs,      
      verbose = 1,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      #steps_per_epoch= (N-len(valIds))//batchSize ,
      steps_per_epoch= N//batchSize ,
      #epochs=int(10000)
      epochs = 10
      )

    print("Saving final model")
    tf.saved_model.save(model, os.path.join(experiment_output_dir,'final/'))    

    print("Done")
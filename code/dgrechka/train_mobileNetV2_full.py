import tensorflow as tf

import sys
import os
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet
from tfDataIngest import tfDataSetParquetAnnotateTrain as tfDsParquetAnnotation
from tfMetrics.macroAveragedRecallForLogits import RecallForLogits
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from models.MobileNetV2 import GetModel

inputDataDir = sys.argv[1]
validationFile = sys.argv[2]
checkpointFile = os.path.join(sys.argv[3],"weights.hdf5")
experiment_output_dir = sys.argv[4]
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
    print("Loading pretrained weights file {0}".format(checkpointFile))
    model.load_weights(checkpointFile)
    print("model weights are loaded")
    
    cnn.trainable = True
    # for layer in cnn.layers:
    #     layer.trainable = True
    
    def catCeFromLogitsDoubled(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)*2.0
    def catCeFromLogits(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)    
    
    model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          #optimizer=tf.keras.optimizers.RMSprop(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss= {
              "root":catCeFromLogitsDoubled,
              "vowel":catCeFromLogits,
              "consonant":catCeFromLogits
          },
          metrics=[[RecallForLogits(168)],[RecallForLogits(11)],[RecallForLogits(7)],[]]
          )
    print("model compiled")

    print(model.summary())

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(experiment_output_dir,'training_log.csv'),append=False)
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=int(10), monitor='val_root_recall',mode='max'),
        # Write TensorBoard logs to `./logs` directory
        # tf.keras.callbacks.TensorBoard(log_dir=experiment_output_dir, histogram_freq = 0, profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(experiment_output_dir,"weights.hdf5"),
            save_best_only=True,
            verbose=True,
            mode='max',
            save_weights_only=True,
            monitor='val_root_recall'),
        tf.keras.callbacks.TerminateOnNaN(),
        csv_logger,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_recall', factor=0.1, patience=5, min_lr=1e-7,mode='max')
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
      epochs = 50
      )    
    print("Done")
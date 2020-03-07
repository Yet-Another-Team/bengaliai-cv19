import tensorflow as tf
import tensorflow_addons as tfa

import sys
import os
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquetP as tfDsParquet
from tfDataIngest import tfDataSetParquetAnnotateTrainP as tfDsParquetAnnotation
from tfMetrics.macroAveragedRecallForLogits import RecallForLogits
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from models.NASNetMobile import GetModel

inputDataDir = sys.argv[1]
validationFile = sys.argv[2]
checkpointFile = os.path.join(sys.argv[3],"weights.hdf5")
experiment_output_dir = sys.argv[4]
dropoutRate = 0.2
batchSize = 64
seed = 313143

print("validation set samples listing: {0}".format(validationFile))

valDf = pd.read_csv(validationFile)
valIds = set(valDf.image_id)
print("{0} samples will be used for validation".format(len(valIds)))

if __name__ == "__main__":    
    tf.random.set_seed(seed+563)

    #tf.compat.v1.disable_eager_execution()
    #tf.config.experimental.set_device_policy('warn')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs: {0}".format(gpus))
    print("Available CPUs: {0}".format(tf.config.experimental.list_physical_devices('CPU')))
    tf.config.experimental.set_memory_growth(gpus[0], True)

    print("Data dir is {0}".format(inputDataDir))
    dataFileNames = glob("{0}/train*.parquet".format(inputDataDir))
    trainLabelsFileName = "{0}/train.csv".format(inputDataDir)

    N = len(pd.read_csv(trainLabelsFileName))
    #N = 3000
    print("There are {0} training samples in total".format(N))

    print("Parquet files count is {0}".format(len(dataFileNames)))
    print("First is {0}".format(dataFileNames[0]))

    def constructAllSamplesDs():
        ds = tfDsParquet.create_parquet_dataset_gen(dataFileNames)
        #ds = tfDsParquet.create_parquet_dataset_gen([dataFileNames[0]])
        ds = tfDsParquetAnnotation.annotate(ds,trainLabelsFileName)  
        return ds  


    # reshaping to match the input shape
    def prepareInput(_,labels,pixels):
        #pixels = tf.cast(pixels, tf.float32)
        root,vowel,consonant = tf.unstack(labels,num=3)
        root = tf.one_hot(root, 168, dtype=tf.float32)
        vowel = tf.one_hot(vowel, 11, dtype=tf.float32)
        consonant = tf.one_hot(consonant, 7, dtype=tf.float32)

        colored = tf.image.grayscale_to_rgb(tf.expand_dims(pixels,-1))
        print('colored shape is {0}'.format(colored.shape))
        pixels = tf.image.resize(colored, [224,224], method='gaussian')
        #HEIGHT = 137
        #WIDTH = 236

        labelsDict = {
            "root": tf.reshape(root,(168,)),
            "vowel": tf.reshape(vowel,(11,)),
            "consonant": tf.reshape(consonant,(7,))
        }
        return pixels, labelsDict

    def shear(sampleId,labels, pixels):        
        #shearFactor = tf.random.uniform([2],minval= -0.2,maxval=0.2,seed=seed)
        shearFactor = tf.random.truncated_normal([2],mean= 0.0,stddev=0.06,seed=seed-234)
        sX = shearFactor[0] # X skew factor
        sY = shearFactor[1] # Y skew factor
        tX = 236*0.5 # image centre X
        tY = 137*0.5 # image centre Y

        # translate -> shear -> translate back

        #  [a0, a1, a2, b0, b1, b2, c0, c1]
        M = [1.0, sX , sX*tY, sY, 1.0, tX*sY, 0.0, 0.0]
        
        #pixInversed = pixels
        pixInversed = 255 - pixels # we need to work with inversed as shearing adds black color on new areas
        shearedInversed = tfa.image.transform(pixInversed, M, interpolation='BILINEAR')
        pixels = 255 - shearedInversed
        #pixels = shearedInversed
        return sampleId,labels,pixels

    def inValidationFilter(ident):            
        identBytes = ident.numpy()
        identStr = identBytes.decode('utf-8')
        return identStr in valIds
    def inValFilter(ident,_dummy_1,_dummy_2):
        return tf.py_function(inValidationFilter, [ident], (tf.bool))
    def inTrainFilter(ident,_dummy_1,_dummy_2):
        return not(tf.py_function(inValidationFilter, [ident], (tf.bool)))


    #with tf.device('/GPU:0'):
    allDs = constructAllSamplesDs()
    allDs = allDs.take(N)
    allDs = allDs.cache()

    print("Caching all DS")
    for element in allDs.as_numpy_iterator():
       ()
    print("all DS cached")

    trDs = allDs.filter(inTrainFilter)


    #print("Caching trDS")
    #for element in trDs.as_numpy_iterator():
    #   ()

    #with tf.device('/GPU:0'):
    #trDs = trDs.cache()
    trDs = trDs.repeat()
    trDs = trDs.shuffle(1024,seed=seed+123678, reshuffle_each_iteration=True)
    trDs = trDs.map(shear,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trDs = trDs.map(prepareInput,num_parallel_calls=tf.data.experimental.AUTOTUNE) # num_parallel_calls=tf.data.experimental.AUTOTUNE
    trDs = trDs.batch(batchSize)
    #trDs = trDs.with_options(options)
    

    #trDs = trDs.prefetch(8)

    #with tf.device('/GPU:0'):
    valDs = allDs.filter(inValFilter)
    #with tf.device('/GPU:0'):
    #valDs = valDs.cache()
    valDs = valDs.map(prepareInput,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valDs = valDs.batch(batchSize)


    #valDs = valDs.prefetch(8)

    print("Training dataSet is {0}".format(trDs))
    print("Validation dataSet is {0}".format(valDs))

    #with tf.device('/GPU:0'):    
    model,cnn = GetModel(dropoutRate, seed+44)
    print("Model constructed")
    model.load_weights(checkpointFile)
    print("model weights are loaded")
    
    cnn.trainable = True
    # for layer in cnn.layers:
    #     layer.trainable = True

    
    def catCeFromLogitsDoubled(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)*2.0
    def catCeFromLogits(y_true, y_pred):        
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)    

    #with tf.device('/GPU:0'):
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
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_recall', factor=0.1, patience=5, min_lr=1e-7,mode='max',verbose = 1)
    ]

    spe = (N-len(valIds))//batchSize 
    vaSteps = len(valIds)//batchSize
    if (len(valIds) % batchSize) != 0:
        vaSteps+=1

    #spe = N//batchSize
    #vaSteps=10

    print("Tt steps per epoch {0}".format(spe))
    print("Va steps per epoch {0}".format(vaSteps))
    #with tf.device('/GPU:0'):
    fitHisotry = model.fit(x = trDs, \
        validation_data = valDs,      
        verbose = 2,
        callbacks=callbacks,
        shuffle=False, # dataset is shuffled explicilty
        steps_per_epoch= spe,
        validation_steps= vaSteps,
        epochs = 75
        )    
    print("Done")
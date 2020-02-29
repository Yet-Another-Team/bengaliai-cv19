import tensorflow as tf

import sys
import os
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet
from tfDataIngest import tfDataSetParquetAnnotateTrain as tfDsParquetAnnotation
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from models.DenseNet121 import GetModel

inputDataDir = sys.argv[1]
validationFile = sys.argv[2]
checkpoint_file = sys.argv[3]
outputFile = sys.argv[4]
dropoutRate = 0.2
batchSize = 8
seed = 1

valDf = pd.read_csv(validationFile)
valIds = set(valDf.image_id)
print("{0} samples will be used for validation".format(len(valIds)))

if __name__ == "__main__":    
    tf.random.set_seed(seed+563)

    print("Data dir is {0}".format(inputDataDir))
    dataFileNames = glob("{0}/train*.parquet".format(inputDataDir))
    trainLabelsFileName = "{0}/train.csv".format(inputDataDir)

    N = len(pd.read_csv(trainLabelsFileName))
    #N = 1000
    print("There are {0} training samples in total".format(N))

    print("Parquet files count is {0}".format(len(dataFileNames)))
    print("First is {0}".format(dataFileNames[0]))


    ds = tfDsParquet.create_parquet_dataset(dataFileNames)
    ds = tfDsParquetAnnotation.annotate(ds,trainLabelsFileName)  
    ds = ds.take(N)
    ds = ds.cache()

    print("Caching all DS")
    for element in tqdm(ds.as_numpy_iterator(),total=N,ascii=True):
       ()

    def inValidationIndicator(ident):            
        identBytes = ident.numpy()
        identStr = identBytes.decode('utf-8')
        res = 0
        if identStr in valIds:
            res = 1
        return res
    def inValIndicator(ident):
        return tf.py_function(inValidationIndicator, [ident], (tf.uint8))
    

    # reshaping to match the input shape
    def prepareModelInput(ident,labels,pixels):
        #pixels = tf.cast(pixels, tf.float32)
        colored = tf.tile(tf.expand_dims(pixels,-1),[1,1,3])

        pixels = tf.image.resize(colored, [224,224], method='gaussian')
        #HEIGHT = 137
        #WIDTH = 236

        #pixels = tf.pad(colored,[[43,44],[0,0],[0,0]])[:,6:230,:]        
        return pixels

    def prepareDescription(ident,labels,pixels):
        valIndicator = inValIndicator(ident)
        return ident,labels, valIndicator

    pixelsDs = ds.map(prepareModelInput)
    pixelsDs = pixelsDs.batch(32)
    descrDs = ds.map(prepareDescription)

    model,cnn = GetModel(dropoutRate, seed+44)
    print("model constructed")
    model.load_weights(checkpoint_file)
    print("model weights are loaded")

    identsArr = np.empty(N, dtype='|U128')
    labelsArr = np.empty((N,3),dtype=np.uint8)
    isValidationArr = np.empty(N,dtype=np.bool)

    bottleneckOut,root,vowel,consonant = model.predict(pixelsDs,verbose=True)
    
    idx = 0
    for element in tqdm(descrDs.as_numpy_iterator(),total=N,ascii=True):
        ident,labels,valIndicator = element
        labelsArr[idx,:] = labels
        identsArr[idx] = ident
        isValidationArr[idx] = valIndicator==1
        idx+=1

    np.savez_compressed(outputFile, features=bottleneckOut, image_id=identsArr, labels=labelsArr, validationIndicator=isValidationArr)
    print("Done")
    
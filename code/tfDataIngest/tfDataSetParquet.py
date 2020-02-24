import pandas as pd
import tensorflow as tf
import os
import numpy as np
from glob import glob

def create_parquet_dataset(parquet_filenames):
    ''' accepts finenames list.
        Produces a dataset of sample batches tuple of tensors (batch x (filename:string); batch x pixelData:3d uint8)).
        Suitable both for training and test set data'''
    HEIGHT = 137
    WIDTH = 236

    def parseParquetTraining(filenameTensor):
        filenameBytes = filenameTensor.numpy()
        filename = filenameBytes.decode('utf-8')
        print("Parsing {0}...".format(filename))
        df = pd.read_parquet(filename)
        N = len(df)
        print("{0} file parsed. {1} lines".format(filename,N))
        sampleIDs = df.image_id
        pixels = df.iloc[:,1:HEIGHT*WIDTH+1].to_numpy(dtype=np.uint8)
        #print("pixels shape is {0}".format(pixels.shape))
        #print("sampleIDs shape is {0}".format(sampleIDs.shape))
        return sampleIDs, pixels
    
    parseParqueteTfOp = lambda fname : tf.py_function(parseParquetTraining, [fname], (tf.string,tf.uint8))

    tf_filenames = tf.constant(parquet_filenames) # A vector of filenames.
    filenames = tf.data.Dataset.from_tensor_slices(tf_filenames) # parsed and split into ID, pixels
    parsed = filenames.map(parseParqueteTfOp) # filenames to tensors
    
    @tf.function
    def reshapePixels(namesBatch,pixelsBatch):
        return namesBatch, tf.reshape(pixelsBatch,[-1,HEIGHT, WIDTH])

    reshaped = parsed.map(reshapePixels)
    # stretching batches
    flattened = reshaped.flat_map(lambda namesBatch,pixelsBatch: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(namesBatch),tf.data.Dataset.from_tensor_slices(pixelsBatch))))
    return flattened

# test app
if __name__ == "__main__":
    files = glob("data/bengaliai-cv19/train*.parquet")
    print("Parquet files {0}".format(len(files)))
    print("First is {0}".format(files[0]))
    ds = create_parquet_dataset(files)

    
    a = 0

    for element in ds.as_numpy_iterator(): 
        print("Iterating...")
        sampleIds,pixels = element
        print(element)
        #print(sampleIds.shape)
        print(pixels.shape)
        a += 1
        if a > 10:
            break
    #print("{0} elements in the dataset".format(len(ds.)))
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
    parsed = filenames.map(parseParqueteTfOp,num_parallel_calls=tf.data.experimental.AUTOTUNE) # filenames to tensors
    
    @tf.function
    def reshapePixels(namesBatch,pixelsBatch):
        return namesBatch, tf.reshape(pixelsBatch,[-1,HEIGHT, WIDTH])

    reshaped = parsed.map(reshapePixels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # stretching batches
    flattened = reshaped.flat_map(lambda namesBatch,pixelsBatch: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(namesBatch),tf.data.Dataset.from_tensor_slices(pixelsBatch))))
    return flattened

def create_parquet_dataset_gen(parquet_filenames):
    ''' accepts finenames list.
        Produces a dataset of sample batches tuple of tensors (batch x (filename:string); batch x pixelData:3d uint8)).
        Suitable both for training and test set data'''
    HEIGHT = 137
    WIDTH = 236

    def gen():
        for filename in parquet_filenames:
            print("Parsing {0}".format(filename))
            df = pd.read_parquet(filename)
            N = len(df)
            print("{0} file parsed. {1} lines".format(filename,N))
            sampleIDs = df.image_id
            pixels = df.iloc[:,1:HEIGHT*WIDTH+1].to_numpy(dtype=np.float32)
            for i in range(0,N):
                #print("pixels single image shape is {0}".format(pixels[i,:].shape))
                yield sampleIDs[i], pixels[i,:]

    ds = tf.data.Dataset.from_generator(gen,(tf.string, tf.float32), (tf.TensorShape([]), tf.TensorShape([HEIGHT * WIDTH])))

    def reshape(sName,pixels):
        return sName,tf.reshape(pixels,[HEIGHT,WIDTH])

    print("ds gen: {0}".format(ds))
    return ds.map(reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)

numpy_cache=dict()

def create_parquet_dataset_numpy(parquet_filenames):
    ''' accepts finenames list.
        Produces a dataset of sample batches tuple of tensors (batch x (filename:string); batch x pixelData:3d uint8)).
        Suitable both for training and test set data'''
    HEIGHT = 137
    WIDTH = 236

    samplesDs = None
    pixelsDs = None

    for filename in parquet_filenames:
        if not(filename in numpy_cache):
            print("Parsing {0}".format(filename))
            df = pd.read_parquet(filename)
            N = len(df)
            print("{0} file parsed. {1} lines".format(filename,N))
            sampleIDs = df.image_id
            pixels = df.iloc[:,1:HEIGHT*WIDTH+1].to_numpy(dtype=np.float32)
            numpy_cache[filename] = (sampleIDs,pixels)
        (sampleIDs,pixels) = numpy_cache[filename]
        curSamplesDs = tf.data.Dataset.from_tensor_slices(sampleIDs)
        curPixelsDs =  tf.data.Dataset.from_tensor_slices(pixels).map(lambda x: tf.reshape(x,[HEIGHT,WIDTH]))
        if samplesDs is None:
            samplesDs = curSamplesDs
            pixelsDs = curPixelsDs
        else:
            samplesDs = samplesDs.concatenate(curSamplesDs)
            pixelsDs = pixelsDs.concatenate(curPixelsDs)   

    ds = tf.data.Dataset.zip((samplesDs,pixelsDs))

    print("ds is: {0}".format(ds))
    return ds


# test app
if __name__ == "__main__":
    files = glob("data/bengaliai-cv19/train*.parquet")
    print("Parquet files {0}".format(len(files)))
    print("First is {0}".format(files[0]))
    ds = create_parquet_dataset_gen(files)

    
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
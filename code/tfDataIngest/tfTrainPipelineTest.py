import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import tfDataSetParquet
import tfDataSetParquetAnnotateTrain

batchSize = 8

if __name__ == "__main__":
    files = glob("data/bengaliai-cv19/train*.parquet")
    print("Parquet files {0}".format(len(files)))
    print("First is {0}".format(files[0]))
    ds = tfDataSetParquet.create_parquet_dataset(files)
    annotatedDs = tfDataSetParquetAnnotateTrain.annotate(ds, "data/bengaliai-cv19/train.csv")

    # reshaping to match the input shape
    def prepareInput(_,labels,pixels):
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

    trDs = annotatedDs.map(prepareInput) #tf.data.experimental.AUTOTUNE
    #trDs = trDs.repeat()
    trDs = trDs.shuffle(8192,seed=123678, reshuffle_each_iteration=True)
    trDs = trDs.batch(batchSize)
    trDs = trDs.prefetch(64)
    
    a = 0

    for element in tqdm(trDs.as_numpy_iterator()): 
        #print("Iterating...")
        labels,pixels = element
        
        #print(sampleIds.shape)
        #print("Pixels shape is {0}".format(pixels.shape))
        #print("Labels shape is {0}".format(labels.shape))
        #a += 1
        #print(element)
        #if a > 10:
        #    break
    #print("{0} elements in the dataset".format(len(ds.)))
    print("Done")
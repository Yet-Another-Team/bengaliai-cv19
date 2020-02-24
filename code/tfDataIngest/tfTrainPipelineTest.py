import tensorflow as tf
import os
import numpy as np
from glob import glob
import tfDataSetParquet
import tfDataSetParquetAnnotateTrain

if __name__ == "__main__":
    files = glob("data/bengaliai-cv19/train*.parquet")
    print("Parquet files {0}".format(len(files)))
    print("First is {0}".format(files[0]))
    ds = tfDataSetParquet.create_parquet_dataset(files)
    annotatedDs = tfDataSetParquetAnnotateTrain.annottate(ds, "data/bengaliai-cv19/train.csv")

    
    a = 0

    for element in annotatedDs.as_numpy_iterator(): 
        print("Iterating...")
        sampleIds,labels,pixels = element
        
        #print(sampleIds.shape)
        print("Pixels shape is {0}".format(pixels.shape))
        print("Labels shape is {0}".format(labels.shape))
        a += 1
        print(element)
        if a > 10:
            break
    #print("{0} elements in the dataset".format(len(ds.)))
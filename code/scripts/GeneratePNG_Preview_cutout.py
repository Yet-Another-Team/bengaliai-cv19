import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import sys
import os
from glob import glob
import png
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet

inputDataDir = sys.argv[1]
outputDir = sys.argv[2]

seed = 3213434

# pixels shape is 137x236
# we will cut outabout 1/4 of heigh = 35x35        
cutoutSize = 32
HEIGHT = 137
WIDTH = 236

# test app
if __name__ == "__main__":
    files = glob(os.path.join(inputDataDir,"train*.parquet"))
    print("Found {0} parquet files in input dir {1}".format(len(files),inputDataDir))
    print("First is {0}".format(files[0]))
    ds = tfDsParquet.create_parquet_dataset([files[0]])

    ds = ds.skip(4).take(1).cache().repeat(300)

    def cutout(pixels):
        # pixels shape is (137,236)
        # we will cut outabout 1/4 of heigh = 35x35

        halfCutoutSize = tf.constant([cutoutSize//2,cutoutSize//2],dtype=tf.int32)
        imBounds = tf.constant([HEIGHT,WIDTH],dtype=tf.int32) # to be broadcasted
        imCentre = tf.constant([HEIGHT//2,WIDTH//2],dtype=tf.int32)
        cutoutDisplacement = tf.cast(tf.random.normal([2],mean=0.0,stddev=30.0,seed=seed), dtype=tf.int32)
        #cutoutDisplacement = [-70,140]
        cutoutCentre = imCentre + cutoutDisplacement
        
        # we need to form the mask tensor with the shape of "pixels"
        patchOrigin = tf.math.minimum(imBounds,tf.math.maximum([0,0], cutoutCentre - halfCutoutSize))
        patchOuterBound = tf.math.maximum([0,0],tf.math.minimum(imBounds, cutoutCentre + halfCutoutSize))

        #patchSize = tf.math.maximum([0,0],patchOuterBound-patchOrigin) # shape (2,)
        patchSize = patchOuterBound-patchOrigin
        patch = tf.ones([patchSize[0], patchSize[1]], tf.int32) # ones PHxPW
        print("Patch shape is {0}".format(patch.shape))
        padBefore = patchOrigin
        padAfter = imBounds-patchOuterBound
        mask = tf.pad(patch,[[padBefore[0],padAfter[0]],[padBefore[1],padAfter[1]]])
        print("Mask shape is {0}".format(mask.shape))

        floatMask = tf.cast(mask, tf.float32)

        blackMaskApplied = tf.math.maximum(0.0,tf.cast(pixels,dtype=tf.float32) - floatMask*255.0)

        grayMaskApplied = blackMaskApplied + floatMask*224.0

        return tf.cast(grayMaskApplied,dtype=tf.uint8)

        

    def cutoutMapper(sampleId, pixels):        
        return sampleId,cutout(pixels)

    ds = ds.map(cutoutMapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(2)

    idx = 0
    for element in ds.as_numpy_iterator(): 
        #print("Iterating...")
        #print("Image generated {0}".format(idx))
        sampleId,pixels = element
        sampleId = sampleId.decode("utf-8")
        fileName = os.path.join(outputDir,"{0}-{1}.png".format(sampleId,idx))
        png.from_array(pixels, mode="L").save(fileName)
        #print("Image saved")
        #print(element)
        #print("sample name is {0}".format(sampleId))
        #print(sampleIds.shape)
        #print(pixels.shape)
        # a += 1
        # if a > 10:
        #     break
        idx += 1
    print("Done")
    #print("{0} elements in the dataset".format(len(ds.)))
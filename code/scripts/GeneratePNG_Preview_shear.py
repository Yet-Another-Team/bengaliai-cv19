import tensorflow as tf
import tensorflow_addons as tfa

import sys
import os
from glob import glob
import png
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet

inputDataDir = sys.argv[1]
outputDir = sys.argv[2]

seed = 3213434

# test app
if __name__ == "__main__":
    files = glob(os.path.join(inputDataDir,"train*.parquet"))
    print("Found {0} parquet files in input dir {1}".format(len(files),inputDataDir))
    print("First is {0}".format(files[0]))
    ds = tfDsParquet.create_parquet_dataset([files[0]])

    ds = ds.skip(4).take(1).repeat(100)

    def shear(sampleId, pixels):
        '''appling skeq transforms'''
        #shearFactor = tf.random.uniform([2],minval= -0.2,maxval=0.2,seed=seed)
        shearFactor = tf.random.truncated_normal([2],mean= 0.0,stddev=0.06,seed=seed)
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
        return sampleId,pixels

    ds = ds.map(shear, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(8)

    idx = 0
    for element in ds.as_numpy_iterator(): 
        #print("Iterating...")
        sampleId,pixels = element
        sampleId = sampleId.decode("utf-8")
        fileName = os.path.join(outputDir,"{0}-{1}.png".format(sampleId,idx))
        png.from_array(pixels, mode="L").save(fileName)
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
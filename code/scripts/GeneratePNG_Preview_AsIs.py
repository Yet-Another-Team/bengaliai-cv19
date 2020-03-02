import tensorflow as tf

import sys
import os
from glob import glob
import png
sys.path.append(os.path.join(__file__,'..','..'))

from tfDataIngest import tfDataSetParquet as tfDsParquet

inputDataDir = sys.argv[1]
outputDir = sys.argv[2]

# test app
if __name__ == "__main__":
    files = glob(os.path.join(inputDataDir,"train*.parquet"))
    print("Found {0} parquet files in input dir {1}".format(len(files),inputDataDir))
    print("First is {0}".format(files[0]))
    ds = tfDsParquet.create_parquet_dataset([files[0]])


    for element in ds.as_numpy_iterator(): 
        #print("Iterating...")
        sampleId,pixels = element
        sampleId = sampleId.decode("utf-8")
        fileName = os.path.join(outputDir,"{0}.png".format(sampleId))
        png.from_array(pixels, mode="L").save(fileName)
        #print(element)
        #print("sample name is {0}".format(sampleId))
        #print(sampleIds.shape)
        #print(pixels.shape)
        # a += 1
        # if a > 10:
        #     break
    print("Done")
    #print("{0} elements in the dataset".format(len(ds.)))
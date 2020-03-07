'''This file contains a function that can supply the parquet dataset with labels'''

import pandas as pd
import tensorflow as tf
import os
import numpy as np
from glob import glob

def annotate(dataset, labelFileName):
    root = dict()
    vowel = dict()
    consonant = dict()
    
    print("Building label index from {0}".format(labelFileName))
    df = pd.read_csv(labelFileName)
    for sample in df.itertuples():
        ident = sample.image_id
        root[ident] = sample.grapheme_root
        vowel[ident] = sample.vowel_diacritic
        consonant[ident] = sample.consonant_diacritic
    print("Built label index")

    def getLabels(ident):
        identBytes = ident.numpy() 
        identStr = identBytes.decode("utf-8")
        #print("identNumpy type is {0}".format(type(identNumpy)))
        #print("identStr is {0}".format(identStr))
        return tf.cast(tf.stack([root[identStr], vowel[identStr], consonant[identStr]]),tf.uint8)

    tfGetLabels = lambda ident : tf.py_function(getLabels, [ident], Tout=tf.uint8)

    @tf.function
    def process(sampleId,pixels):
        return sampleId, tfGetLabels(sampleId), pixels
    
    return dataset.map(process,num_parallel_calls=tf.data.experimental.AUTOTUNE)
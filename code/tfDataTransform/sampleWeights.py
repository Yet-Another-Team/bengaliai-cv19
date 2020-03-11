import tensorflow as tf

def GetClassWeights(stats):
    '''stats - 1D tensor. Values proportional to the dataset occurrence ratio of the classes, used in truthOneHot'''
    maxOccurClassIdx = tf.argmax(stats)
    statsRecip = 1.0 / stats
    statsRecipTotal = tf.math.reduce_sum(statsRecip)
    weights = statsRecip / statsRecipTotal
    normalizeFactor = 1.0 / weights[maxOccurClassIdx]
    weightsNormalized = weights * normalizeFactor # the most abundant class will have weight 1.0, more rare classess will have high weights proportional to rarity
    return weightsNormalized

def GetSampleWeightsFromClassWeights(classWeights,truthOneHot):
    '''classWeights - 1D tensor with shape (Batch,). truthOneHot - 2D tensor with shape (Batch,Classes)'''
    # stats shape is (Classes)
    # truth shape is (Batch x Classes)
    return tf.reduce_sum(classWeights * truthOneHot, axis = -1) # output shape is (Batch,)


def GetSampleWeightsFromClassWeightsBengali(rootClassWeights,rootTruthOneHot,vowelClassWeights,vowelTruthOneHot,consonantClassWeights,consonantTruthOneHot):
    root = GetSampleWeightsFromClassWeights(rootClassWeights,rootTruthOneHot)
    vowel = GetSampleWeightsFromClassWeights(vowelClassWeights, vowelTruthOneHot)
    consonant = GetSampleWeightsFromClassWeights(consonantClassWeights,consonantTruthOneHot)
    return root*vowel*consonant
    
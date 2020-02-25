import tensorflow as tf

def GetModel(dropoutRate):
    imageSize = 224    
    modelInput = tf.keras.Input(shape=(imageSize,imageSize,3),name="input")  
    modelInputPreprocessed = tf.keras.applications.densenet.preprocess_input(modelInput)
    model = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False, pooling='None')
    modelOutput = model(modelInputPreprocessed)
    
    modelOutputStretched = tf.keras.layers.Reshape((7*7*1024,))(modelOutput)

    modelOutputDo = tf.keras.layers.Dropout(dropoutRate)(modelOutputStretched)

    rootDenseOutput = tf.keras.layers.Dense(168, name="root")(modelOutputDo)
    vowelDenseOutput = tf.keras.layers.Dense(11, name="vowel")(modelOutputDo)
    consonantDenseOutput = tf.keras.layers.Dense(7, name="consonant")(modelOutputDo)
    return tf.keras.Model(name="BengaliDenseNet121",inputs=modelInput, outputs=(rootDenseOutput, vowelDenseOutput, consonantDenseOutput))
    
    
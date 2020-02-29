import tensorflow as tf

def GetModel(dropoutRate,seed):
    imageSize = 224    
    modelInput = tf.keras.Input(shape=(imageSize,imageSize,3),name="input")  
    modelInputPreprocessed = tf.keras.applications.densenet.preprocess_input(modelInput)
    model = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False, pooling=None)
    model.trainable = False
    modelOutput = model(modelInputPreprocessed)
    #print("Model is {0}".format(modelOutput))
    
    modelOutputDo = tf.keras.layers.Dropout(dropoutRate,noise_shape=(1, 1, 1024),seed= seed+1)(modelOutput)
    modelSqueezed = tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation="selu",padding='valid')(modelOutputDo) #1024 -> 128 channels
    modelSqueezedStretched = tf.keras.layers.Reshape((7*7*128,),input_shape=(7,7,128))(modelSqueezed)
    modelSqueezedDo = tf.keras.layers.Dropout(dropoutRate,seed=seed+2)(modelSqueezedStretched)
    bottleNeck = tf.keras.layers.Dense(512,activation="selu")(modelSqueezedDo)
    bottleNeckDo = tf.keras.layers.Dropout(dropoutRate,seed=seed+3,name='bottleneckOut')(bottleNeck)
    rootDenseOutput = tf.keras.layers.Dense(168, name="root")(bottleNeckDo)
    vowelDenseOutput = tf.keras.layers.Dense(11, name="vowel")(bottleNeckDo)
    consonantDenseOutput = tf.keras.layers.Dense(7, name="consonant")(bottleNeckDo)
    return tf.keras.Model(name="BengaliDenseNet121",inputs=modelInput, outputs=(bottleNeckDo,rootDenseOutput, vowelDenseOutput, consonantDenseOutput)),model
    
    
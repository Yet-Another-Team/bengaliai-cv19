import tensorflow as tf

def GetModel(dropoutRate,seed):
    imageSize = 224    
    modelInput = tf.keras.Input(shape=(imageSize,imageSize,3),name="input")  
    modelInputPreprocessed = tf.keras.applications.nasnet.preprocess_input(modelInput)
    model = tf.keras.applications.NASNetMobile(weights='imagenet',include_top=False, pooling='max', input_shape=(imageSize,imageSize,3))
    model.trainable = False
    modelOutput = model(modelInputPreprocessed)
    print("Model output shape is {0}".format(modelOutput.shape))
    
    modelOutputDo = tf.keras.layers.Dropout(dropoutRate,seed= seed+1)(modelOutput) # (None , 1056)
    # modelOutputDo = tf.keras.layers.Dropout(dropoutRate,noise_shape=(1, 1, 1280),seed= seed+1)(modelOutput)
    # modelSqueezed = tf.keras.layers.Conv2D(filters=128,kernel_size=1,activation="selu",padding='valid')(modelOutputDo) #1024 -> 128 channels
    # modelSqueezedStretched = tf.keras.layers.Reshape((7*7*128,),input_shape=(7,7,128))(modelSqueezed)
    # modelSqueezedDo = tf.keras.layers.Dropout(dropoutRate,seed=seed+2)(modelSqueezedStretched)

    bottleNeck = tf.keras.layers.Dense(512,activation="selu")(modelOutputDo)
    bottleNeckDo = tf.keras.layers.Dropout(dropoutRate,seed=seed+3,name='bottleneckOut')(bottleNeck)
    rootDenseOutput = tf.keras.layers.Dense(168, name="root")(bottleNeckDo)
    vowelDenseOutput = tf.keras.layers.Dense(11, name="vowel")(bottleNeckDo)
    consonantDenseOutput = tf.keras.layers.Dense(7, name="consonant")(bottleNeckDo)
    return tf.keras.Model(name="BengaliNASNetMobile",inputs=modelInput, outputs=(rootDenseOutput, vowelDenseOutput, consonantDenseOutput, bottleNeckDo)),model
    
    
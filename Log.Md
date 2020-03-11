# Research Log

## 2020-02-28

DenseNet121 has too high capacity for the problem (retraining of only bottlneck, regularized with dropout, resulted in overfitting). Also it takes ~15 minutes on GeForce 1050 Ti to predict bottleneck layer outputs for entire training set. To reduce both capacity and prediction time we can use pimplier networks (e.g. MobileNetV2 or NASNetMobile) as a feature extractor.

## 2020-03-1

### Transfer learning (part 1): imagenet trained cnn fixed, bottleneck training

MobileNetV2 :
Total params: 3,009,274
Trainable params: 751,290
Non-trainable params: 2,257,984

NASNetMobile
Total params: 4,906,318
Trainable params: 636,602
Non-trainable params: 4,269,716

### Transfer learning (part 2): Continuing to train will all parameters trainable

## 2020-03-10

Experiment 4 (NASNet full transfer) showed better metrics than 3 (MobileNet full transfer)
Paired T-test for validation macro recall showed significance for all tree classifications separately.
See experiment analysis.


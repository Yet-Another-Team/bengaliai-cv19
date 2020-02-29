# Research Log

## 2020-02-28

DenseNet121 has too high capacity for the problem (retraining of only bottlneck, regularized with dropout, resulted in overfitting). Also it takes ~15 minutes on GeForce 1050 Ti to predict bottleneck layer outputs for entire training set. To reduce both capacity and prediction time we can use pimplier networks (e.g. MobileNetV2 or NASNetMobile) as a feature extractor.

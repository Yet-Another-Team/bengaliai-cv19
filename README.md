# Bengali.AI Kaggle challenge research made with Alpheus

This repo contains a ML research of solving [Bengali.AI Handwritten Grapheme Classification Kaggle Challenge](https://www.kaggle.com/c/bengaliai-cv19) with [Alpheus computational experiment management tool](https://github.com/itislab/alpheus).

This readme primarily is a research log containing experiments motivation and results. It is also can be considered as a demo for the Alpheus usage during the whole ML research process.

# Day 1

## Organizing initial data, EDA, preparing cross validation splits

Initializing the alphues directory within git controlled dir
`alpheus init`

### Downloading and extracting the data

We can then create the first Alpheus artefact - kaggle data

We do so by building a method that produces the single artefact (zip file) by executing CLI command of kaggle API.

```shell
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus build -o bengaliai-cv19.zip "kaggle competitions download -c bengaliai-cv19"
```

Now we can add another method that extracts the downloaded data to some dir

```shell
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus build -d ..\code\scripts\unzip.ps1 -d bengaliai-cv19.zip -o .\bengaliai-cv19\ "powershell.exe -File $in1 $in2 $out1"
```

 > Notice: We both used script to execute and the file to unzip as inputs (supplied by -d parameters).

 > Notice: We specified output with trailing backslash. It gives alpheus the clue that the output will be a directory rather than single file.

Then we can actually download and extract the data by asking to compute final artefact.
Alpheus will compute the whole method chain: first download then extract.

```shell
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus compute bengaliai-cv19\
```

We need to establish *baseline model* training pipeline to track the performance gain during the experiments.

But even before that we need to create reproducible training/validation split so we have metrics for the baseline and other models comparable.

The evaluation metric for this competition is hierarchical macro-averaged recall. That means that false negatives are heavily punished for the rare classes.

We should check the for classes imbalance as it can affect the training/validation split process.

### Analyzing prior class distributions

Now we can analyse the class distributions in the training set

```shell
(venv-gpu) C:\ML\bengaliai-cv19\initialDataAnalysis>alpheus build -d ..\code\scripts\EvaluateClassDensity.R -d ..\data\bengaliai-cv19\ -o GraphemeRootPriorProbs.csv -o VowelDiacriticPriorProbs.csv -o ConsonantDiacriticPriorProbs.csv "RScript $in1 $in2\train.csv $out1 $out2 $out3"
(venv-gpu) C:\ML\bengaliai-cv19\initialDataAnalysis>alpheus compute GraphemeRootPriorProbs.csv.alph
```

> Notice: We run alphues from different working directories. Relevant paths used in commands account for this.

> Notice: We set the whole extracted initial data directory as a dependency (input), but pass only one file from it the command. e.g. $in2\train.csv

Now by looking at the output files we can see that prior probabilities varies between 0.0006 (e.g. for classes 77 and 33) to 0.028 (class 72) which is **two decimal orders imbalance**.

For the Vowel Diacritic the prior probs varies between 0.017 and 0.206, so **one decimal order imbalance**.
For Consonant Diacritic the prior probs varies between 0.003 and 0.62 which is huge imbalance of **two decimal orders imbalance**

We can do K-fold cross validation (as a bonus we can ensample models for difference folds at the end).

As we have class imbalance we can split the folds keeping the ratio of classes the same both in training and validation sets. Prior classes distributions will be preserved.

I'll do that by outputting different files for different folds, each file will contain the sample IDs of the validation set in the corresponding fold.

I'll use the F# script as I'm more confident in F#, but you can do it in python or in any other CLI scripting tool.

```shell
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus build -d ..\packages\FSharp.Compiler.Tools\tools\fsiAnyCpu.exe -d ..\code\scripts\GenerateCVSplits.fsx -d bengaliai-cv19\ -o .\5foldCvSplits\*.val_ids.csv -o .\5foldCvSplits\*.root_stats.csv -o .\5foldCvSplits\*.vowel_stats.csv -o .\5foldCvSplits\*.consonant_stats.csv "$in1 $in2 $in3\train.csv $out1 $out2 $out3 $out4 5"
```

 > Notice: We produce the outputs with asteriks (*) in the path. It means that the output contains several files matching the pattern. That's called vector artefact.

 > Notice: We fixed number of folds equal to 5 (script parameter) in the build command above.

```shell
(venv-gpu) C:\ML\bengaliai-cv19>alpheus -v verbose build -rg gpu -d code\dgrechka\train_mobileNetV2_bottleneck.py -d data\bengaliai-cv19\ -d data\5foldCvSplits\*.val_ids.csv -d code\models\MobileNetV2.py -o experiment_outputs\dgrechka_1_mobileNetV2_bottleneck\*\ "python $in1 $in2 $in3 $out1"
```

```shell
(venv-gpu) C:\ML\bengaliai-cv19>alpheus -v verbose build -rg gpu -d code\dgrechka\train_mobileNetV2_full.py -d data\bengaliai-cv19\ -d data\5foldCvSplits\*.val_ids.csv -d experiment_outputs\dgrechka_1_mobileNetV2_bottleneck\*\ -d code\tfDataIngest\tfDataSetParquet.py -d code\tfDataIngest\tfDataSetParquetAnnotateTrain.py -d code\models\MobileNetV2.py -d code\tfMetrics\macroAveragedRecallForLogits.py -o experiment_outputs\dgrechka_2_mobileNetV2_full_transfer_learning\*\ "python $in1 $in2 $in3 $in4 $out1"
```

```shell
alpheus -v verbose build -rg gpu -d code\dgrechka\train_NASNetMobile_full_shear_cls_weights.py -d data\bengaliai-cv19\ -d data\5foldCvSplits\*.val_ids.csv -d initialDataAnalysis\GraphemeRootPriorProbs.csv -d initialDataAnalysis\VowelDiacriticPriorProbs.csv -d initialDataAnalysis\ConsonantDiacriticPriorProbs.csv -d experiment_outputs\dgrechka_2_NASNetMobile_bottleneck\*\ -o experiment_outputs\dgrechka_6_NASNetMobile_full_aug_sample_weights\*\ -d code\models\NASNetMobile.py -d code\tfDataIngest\tfDataSetParquetAnnotateTrainP.py -d code\tfDataIngest\tfDataSetParquetP_uint8.py -d code\tfMetrics\macroAveragedRecallForLogits.py -d code\tfDataTransform\sampleWeights.py "python $in1 $in2 $in3 $in4 $in5 $in6 $in7 $out1"
```

```shell
alpheus build -d code\scripts\TrainResultsPairedTTest.R -d experiment_outputs\dgrechka_1_mobileNetV2_bottleneck\ -d experiment_outputs\dgrechka_2_NASNetMobile_bottleneck\ -o experiment_analysis\dgrechka_1_vs_2_val_root_recall_ptt.txt RScript $in1 $in2 $in3 val_root_recall $out1
```

```shell
alpheus -v verbose build -rg memory -d code\dgrechka\predict_bottleneck_4.py -d data\bengaliai-cv19\ -d data\5foldCvSplits\*.val_ids.csv -d experiment_outputs\dgrechka_4_NASNetMobile_full_transfer_learning\*\weights.hdf5 -o data\bottlenecks\dgrechka_4\*.extracted_features.npz -d code\tfDataIngest\tfDataSetParquet.py -d code\tfDataIngest\tfDataSetParquetAnnotateTrain.py -d code\models\NASNetMobile.py "python $in1 $in2 $in3 $in4 $out1"
```

```shell
alpheus -v verbose build -rg cpu -d code\trainCaretXgboostTree.R -d data\bottlenecks\dgrechka_4\*.extracted_features.npz -o experiment_outputs\dgrechka_10_xgboost_train_bn_dgrechka_4\*\ "RScript $in1 $in2 $out1"
```
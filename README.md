### Bengali.AI Kaggle challenge research made with Alpheus

This repo contains a ML research of solving [Bengali.AI Handwritten Grapheme Classification Kaggle Challenge](https://www.kaggle.com/c/bengaliai-cv19) with [Alpheus computational experiment management tool](https://github.com/itislab/alpheus).

This readme primarily is a research log containing experiments motivation and results. It is also can be considered as a demo for the Alpheus usage during the whole ML research process.

# Day 1

## Organizing initial data, preparing cross validation splits

Initializaing the alphues directory within git controlled dir
`alpheus init`

We can then create the first Alpheus artefact - kaggle data

We do so by building a method that produces the single artefact (zip file) by executing CLI command of kaggle API.

```{shell}
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus build -o bengaliai-cv19.zip "kaggle competitions download -c bengaliai-cv19"
```

Now we can add another method that extracts the downloaded data to some dir

```{shell}
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus build -d ..\code\scripts\unzip.ps1 -d bengaliai-cv19.zip -o .\bengaliai-cv19\ "powershell.exe -File $in1 $in2 $out1"
```

Note that we both used script to execute and the file to unzip as inputs (supplied by -d paramters).
Also note that we specified output with trailing backslash. It gives alpheus the clue that the output will be a directory rather than single file.

Then we can actually download and extract the data by asking to compute final artefact.
Alpheus will compute the whole method chain: first download then extract.

```{shell}
(venv-gpu) C:\ML\bengaliai-cv19\data>alpheus compute bengaliai-cv19\
```

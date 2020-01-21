# AidMe

## Installation

You will need Conda (prefer miniconda, see [here](https://conda.io/docs/user-guide/install/index.html) for installation) to create a virtual environment and install the required packages.
Run setup.sh to create the virtual environment, install dependencies and download some necessary data.

```
./setup.sh
```

*OR* run the following command for a step by step process.
To install a virtual environment, run :

```shell script
conda env create -f requirements.yml
```

Once the virtual environment is set up, activate it:
 ```shell script
conda activate aidme
```

You also need to download spacy english corpus. 

```shell script
python -m spacy download en
```

Finally create a folder data/word_embedding and unzip Paragram_sl999 in it (download from http://www.cs.cmu.edu/~jwieting/)


## Reproducing demos

To be continued
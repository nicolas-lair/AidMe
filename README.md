# AidMe
This the code of the paper User-in-the-loop adaptive intent detection for instructable digital assistant that was published at IUI'20 (Intelligent User Interface)

The article is available here https://dl.acm.org/doi/abs/10.1145/3377325.3377490 or on arxiv

## Installation

### Requirements: Conda and gcc
- You will need Conda (prefer miniconda, see [here](https://docs.conda.io/en/latest/miniconda.html) for installation) to create a virtual environment and install the required packages.
If you need to install conda, you will have to close the terminal window and reopen it before proceeding to the following instructions.

- You will also need gcc on Linux (`sudo apt-get install gcc`).

### Dependencies

Creata a virtual environment and install dependencies:

```shell script
conda env create -f requirements.yml
```

Activate the environment

```shell script
conda activate aidme
```

You can then run setup.sh to download some necessary data :

```
./setup.sh
```

## Reproducing demos

To be continued


## Citation : 

@inproceedings{10.1145/3377325.3377490,
author = {Lair, Nicolas and Delgrange, Clement and Mugisha, David and Dussoux, Jean-Michel and Oudeyer, Pierre-Yves and Dominey, Peter Ford},
title = {User-in-the-Loop Adaptive Intent Detection for Instructable Digital Assistant},
year = {2020},
isbn = {9781450371186},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3377325.3377490},
doi = {10.1145/3377325.3377490},
booktitle = {Proceedings of the 25th International Conference on Intelligent User Interfaces},
pages = {116–127},
numpages = {12},
keywords = {digital assistant, user-in-the-loop, learning by interaction, intent detection, natural language processing, multi-domain},
location = {Cagliari, Italy},
series = {IUI ’20}
}
  

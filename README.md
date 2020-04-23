# AidMe
This the code of the paper  

available here https://dl.acm.org/doi/abs/10.1145/3377325.3377490 or on arxiv

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

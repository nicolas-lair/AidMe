# Create virtual environment and install dependencies
conda env create -f requirements.yml

# Load spacy model
conda activate aidme
python -m spacy download en

# Load paragram_300_sl999 from google drive using this gist from beliys at https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
mkdir data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B9w48e1rj-MOck1fRGxaZW1LU2M' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B9w48e1rj-MOck1fRGxaZW1LU2M" -O paragram_300_sl999.zip && rm -rf /tmp/cookies.txt
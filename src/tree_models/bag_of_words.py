import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.data import get_tokenizer


class TfidfBagOfWords(TfidfVectorizer):
    def __init__(self, tokenizer=get_tokenizer('spacy'), **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.corpus_size = 0

    def fit(self, raw_doc, **kwargs):
        self.corpus_size = len(raw_doc)
        super().fit(raw_doc, **kwargs)

    def get_idf_dict(self):
        return dict(zip(self.get_feature_names(), self.idf_))

    def save(self, path):
        self.tokenizer = None
        joblib.dump(self, path)


def load_bow(path, tokenizer=get_tokenizer('spacy')):
    vectorizer = joblib.load(path)
    vectorizer.set_params(tokenizer=tokenizer)
    return vectorizer


def train_tfidfbow(df, output_folder, name='tf_idf.pkl', save=True):
    raw_document = list(df.sent1) + list(df.sent2)
    tfidf_bow = TfidfBagOfWords(lowercase=False)
    tfidf_bow.fit(raw_document)
    if save:
        tfidf_bow.save(os.path.join(output_folder, name))

    idf_dict = tfidf_bow.get_idf_dict()
    return tfidf_bow, idf_dict

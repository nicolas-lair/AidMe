import os

import joblib
import pandas as pd
from torchtext.data import BucketIterator, Iterator

import loader
import logger as lg
from Tabular_dts import build_vocab
from model import XGBoost, RandomForest, NeuralModel, Infersent_model

logger = lg.logging.getLogger(name='train')
logger.setLevel(lg.logging.DEBUG)


def create_models_list(model_params, sentence_field, use_gpu):
    def get_model(key, value):
        if 'xgboost' in key:
            model_class = XGBoost(**value)
        elif 'forest' in key:
            model_class = RandomForest(**value)
        elif 'binary' in key:
            model_class = NeuralModel(sentence_field=sentence_field, use_gpu=use_gpu, **value)
        elif 'multi' in key:
            model_class = NeuralModel(sentence_field=sentence_field, use_gpu=use_gpu, **value)
        elif 'infersent' in key:
            model_class = Infersent_model()
        else:
            raise NotImplementedError
        return model_class

    models_list = []
    for k, v in model_params.items():
        models_list.append(get_model(k, v))
    return models_list


def prepare_data_nn(path, headers, sentence_field, dts_loader, train, embedding_folder=None, embedding=None, ind=None,
                    **kwargs):
    if train:
        return prepare_train_data_nn(path, headers, sentence_field, embedding_folder=embedding_folder,
                                     embedding=embedding, dts_loader=dts_loader, ind=ind)
    else:
        return prepare_test_data_nn(path, headers, sentence_field, dts_loader=dts_loader, ind=ind)


def prepare_train_data_nn(path, headers, sentence_field, embedding_folder, embedding, dts_loader=loader.csv_to_dataset,
                          ind=None):
    """

    :param embedding:
    :param embedding_folder:
    :param sentence_field:
    :param headers:
    :param path:
    :param dts_loader:
    :param args:
    :param ind:
    :return:
    """
    train_data = dts_loader(path, headers, keep_ind=ind, sentence_field=sentence_field)
    # TODO choose embedding
    build_vocab(train_data, embedding_folder, embedding)
    train_iter = BucketIterator(train_data, 32, repeat=False, sort_key=(lambda x: len(x.sent1) + len(x.sent2)))
    return train_iter


def prepare_test_data_nn(path, headers, sentence_field, dts_loader=loader.csv_to_dataset, ind=None):
    """

    :param path:
    :param dts_loader:
    :param headers:
    :param sentence_field:
    :param ind:
    :return:
    """
    test_data = dts_loader(path, headers, keep_ind=ind, sentence_field=sentence_field)
    test_iter = Iterator(test_data, 1, repeat=False, shuffle=False, train=False, sort=False)
    return test_iter


def load_features(data_folder, args, train_or_test):
    features = []
    for f in ['pair_features', 'bow_features', 'wef']:
        if args[f'use_{f}']:
            logger.info(f'Loading {f}')
            features.append(pd.read_csv(os.path.join(data_folder, f'{train_or_test}_{f}.tsv'), sep='\t'))
    try:
        df_features = pd.concat(features, axis=1)
    except ValueError:
        df_features = None

    labels = joblib.load(os.path.join(data_folder, f'{train_or_test}_scores.pkl'))

    return df_features, labels


def prepare_data_for_infersent(path, headers, keep_ind=None):
    df = loader.csv_to_df(path, headers, keep_ind)
    sentence1 = list(df[headers[0]])
    sentence2 = list(df[headers[1]])
    return sentence1, sentence2

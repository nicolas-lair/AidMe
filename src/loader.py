import csv
import os

import joblib
import pandas as pd
import torch
from torchtext import data
from torchtext.data import Example, Dataset, Field

import model
from Tabular_dts import TabularDataset
from neural_models.models import BinModel, MultiClassModel


########################################################################################################################
# # # Load data to a DATASET : --> dts_loader for compute_features function in preprocessing module
# # # - from a csv/tsv file
# # # - from a list of list

def create_list_fields(headers, sentence_field=None):
    if sentence_field is None:
        sentence_field = Field(sequential=True, tokenize=data.get_tokenizer('spacy'), lower=True)

    score_field = Field(sequential=False, preprocessing=float, dtype=torch.float, use_vocab=False)

    list_fields = []
    for s in headers:
        if s == 'score':
            field = ('score', score_field)
        elif s in ['sent1', 'sent2']:
            field = (s, sentence_field)
        else:
            field = (s, None)
        list_fields.append(field)
    return score_field, sentence_field, list_fields


def get_filter(keep_ind):
    if keep_ind is not None:
        a = 0

        def filter_(_):
            nonlocal a
            keep = a in keep_ind
            a += 1
            return keep
    else:
        filter_ = None
    return filter_


def csv_to_dataset(path, headers, keep_ind=None, format='tsv', sentence_field=None):
    filter_ = get_filter(keep_ind)
    _, _, list_fields = create_list_fields(headers, sentence_field=sentence_field)
    train = TabularDataset(path=path, format=format, fields=list_fields, filter_pred=filter_)
    return train


def list_to_dataset(data_list, headers, keep_ind=None, sentence_field=None):
    filter_ = get_filter(keep_ind)
    _, _, list_fields = create_list_fields(headers, sentence_field=sentence_field)
    # example_list = [Example.fromlist([s[0], s[1]], list_fields) for s in data]
    example_list = [Example.fromlist(s, list_fields) for s in data_list]
    dts = Dataset(examples=example_list, fields=list_fields, filter_pred=filter_)
    return dts


########################################################################################################################
# # # Load data to a pandas dataframe : --> df_loader for compute_features function in preprocessing module
# # # - from a csv/tsv file
# # # - from a list of list

def csv_to_df(path, headers, keep_ind=None):
    df = pd.read_csv(path, header=None, quoting=csv.QUOTE_NONE, sep='\t',
                     usecols=range(len(headers)))
    df.columns = headers
    if keep_ind is not None:
        df = df.iloc[keep_ind]
    return df


def list_to_df(data_list, headers, keep_ind=None):
    df = pd.DataFrame(data_list)
    df.columns = headers
    if keep_ind is not None:
        df = df.iloc[keep_ind]
    return df


########################################################################################################################
# # # Other loaders
# # #       - tfidf
# # #       - sentence_field
# # #       - model

def load_tfidfbow(folder, name='tf_idf.pkl', tokenizer=data.get_tokenizer('spacy')):
    vectorizer = joblib.load(os.path.join(folder, name))
    vectorizer.set_params(tokenizer=tokenizer)
    # tfidf_bow = load_bow(os.path.join(args['output'], args['model_tfidf']))
    idf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    return vectorizer, idf_dict


def load_sentence_field(path):
    sentence_field = torch.load(path)
    sentence_field.dtype = torch.int64
    sentence_field.tokenize = data.get_tokenizer('spacy')
    return sentence_field


def load_ensemble_model(model_path, sentence_field, use_gpu=False):
    bin_model = BinModel(sentence_field, 300, use_gpu=use_gpu)  # TODO check size of model
    mc_model = MultiClassModel(sentence_field, 300, use_gpu=use_gpu)
    models_list = [model.XGBoost(), model.RandomForest(), model.NeuralModel(bin_model, name='bin_nn'),
                   model.NeuralModel(mc_model, name='mc_nn')]
    m = model.EnsembleModel(models_list)
    m.load(model_path)
    return m


# def load_semeval_data(data_path, train, val=None, test=None):
#     """
#     Load data from semeval DATASET into a tabulardataset
#     :param data_path: data folder
#     :param train: name of train DATASET file
#     :param val: name of val DATASET file
#     :param test: name of test DATASET file
#     :return:
#     """
#
#     SCORE = data.Field(sequential=False, preprocessing=float, dtype=torch.float, use_vocab=False)
#     SENT = data.Field(sequential=True, tokenize=data.get_tokenizer('spacy'), lower=True)
#
#     list_fields = [('type', None), ('col', None), ('origin', None), ('ind', None),
#                    ('score', SCORE), ('sent1', SENT), ('sent2', SENT),
#                    ('source', None), ('add', None)]
#
#     train, val, test = TabularDataset.splits(
#         path=data_path,
#         train=train,
#         validation=val,
#         test=test,
#         format='tsv',
#         fields=list_fields)
#
#     return train, val, test

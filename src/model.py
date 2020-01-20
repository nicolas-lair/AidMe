#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import joblib
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import nn

import logger as lg
from neural_models.eval import compute_predictions
from neural_models.models import BinModel, MultiClassModel
from neural_models.train import train_model
from tree_models.regression_algorithms import train_randomforest, train_xgboost_scikit

logger = lg.logging.getLogger(name='ensemble_model')
logger.setLevel(lg.logging.DEBUG)


def ensemble_average(scores):
    """Module that combines the final scores by averaging them.

    Args:
        scores(np.ndarray): n samples * m scores matrix
    
    Returns:
        np.ndarray: Array with means
    """
    return scores.mean(axis=1)


def pearsonr_value(x, y):
    """
    Compute the pearson correlation value between two sets of values
    :param x: array-like
    :param y: array-like
    :return: float
    """
    return stats.pearsonr(x, y)[0]


class BaseModel:
    """
    Semi Abstract Class implementing basic functions of any models:
    - fit : fit a model
    - predict : predict labels after fitting
    - evaluate : evaluate a metric given a test DATASET and test labels
    """

    def __init__(self, model=None, name=None):
        """

        :param model: any initialized model, may be unfitted
        :param name: nam of the model, necessary for ensemble model
        """
        self._model = model
        self.name = name

    def fit(self, **kwargs):
        pass

    def predict(self, data):
        """

        :param data: data
        :return:
        """
        return self._model.predict(data)

    def evaluate(self, data, y_true, metrics=pearsonr_value):
        """

        :param data: data
        :param y_true: labels
        :param metrics: metric function , callable
        :return:
        """
        y_pred = self.predict(data)
        return metrics(y_pred, y_true)

    def save(self, path):
        joblib.dump(self._model, path)

    def load(self, path):
        self._model = joblib.load(path)


class RandomForest(BaseModel):
    """
    Implement RandomForest
    dtype attribute : nature of the model, useful to discriminate the nature of input data needed when the model is
    inside an ensemble model
    """

    def __init__(self, max_depth):
        super().__init__(name='random_forest')
        self.dtype = 'reg'
        self.max_depth = max_depth

    def fit(self, data, y, **kwargs):
        self._model = train_randomforest(data, y, max_depth=self.max_depth, **kwargs)

    # def predict(self, data):
    #     """
    #     Random Forest do not accept nan values
    #     :param data: data
    #     :return: array
    #     """
    #     non_nan_data = np.nan_to_num(data)
    #     return super().predict(non_nan_data)


class XGBoost(BaseModel):
    """
    Implement XGBoost model
    """

    def __init__(self, max_depth):
        super().__init__(name='xgboost')
        self.dtype = 'reg'
        self.max_depth = max_depth

    def fit(self, data, y, **kwargs):
        self._model = train_xgboost_scikit(data, y, max_depth=self.max_depth, **kwargs)


class NeuralModel(BaseModel):
    """
    Wrapper to neural model defined in the neural model folder
    """

    def __init__(self, name, sentence_field, output_size, emb_dim=300, label_range=(0, 5),
                 hidden_activation_layer=nn.ReLU(), optimizer=torch.optim.Adam, loss_func=nn.KLDivLoss(), epochs=5,
                 use_gpu=True, verbose=False,
                 ):

        """
        :param model: initialized model of class BinModel or MultiClassModel
        :param optimizer: function from torch.optim
        """
        if name == 'binary_nn':
            model = BinModel(sentence_field, emb_dim, label_range=label_range, output_size=output_size,
                             hidden_activation_layer=hidden_activation_layer, use_gpu=use_gpu)
        elif name == 'multi_class_nn':
            model = MultiClassModel(sentence_field, emb_dim, output_size=output_size, label_range=label_range,
                                    hidden_activation_layer=hidden_activation_layer, use_gpu=use_gpu)
        else:
            raise NotImplementedError

        super().__init__(model=model, name=name)
        self.dtype = 'nn'
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, data, val_iter=None, y=None, **kwargs):
        """

        :param data: data iterator
        :param val_iter: validation iterator
        :param y: unused, simply for compatibility
        :param kwargs:
        :return:
        """
        optimizer = self.optimizer(self._model.parameters(), lr=1e-2)
        return train_model(model=self._model, train_iter=data, val_iter=val_iter, optimizer=optimizer,
                           loss_func=self.loss_func, epochs=self.epochs, verbose=self.verbose, **kwargs)

    def predict(self, data):
        """
        Give an iterator, predict the scores
        :param data:
        :return: predictions as numpy array (+ scores as numpy array)
        """
        predictions, _ = compute_predictions(self._model, data)
        return predictions.numpy()

    def evaluate(self, data, y_true, metrics=pearsonr_value):
        """
        Given a data iterator and labels, evaluate the performance of the model.
        The data iterator and the labels should be aligned.
        If torchtext.data.Iterator is used, set shuffle parameter to False.
        You should use src.demo_utils.prepare_test_data_nn

        :param data: data iterator without shuffling
        :param y_true: true labels, array
        :param metrics: callable
        :return: float
        """
        predictions = self.predict(data)
        return metrics(predictions, y_true)

    def update_embedding_layer(self, vocab):
        self._model.update_embedding_layer(vocab)

    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()


class Infersent_model(BaseModel):
    def __init__(self, label_range=(0, 5)):
        self.dtype = 'txt'
        self.min_label, self.max_label = label_range
        self.label_range = self.max_label - self.min_label

        from InferSent.models import InferSent
        MODEL_PATH = 'InferSent/encoder/infersent2.pkl'

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0,
                        'version': 2}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = 'InferSent/fastText/crawl-300d-2M.vec'
        infersent.set_w2v_path(W2V_PATH)
        super().__init__(name='infersent', model=infersent)

    def fit(self, data, **kwargs):
        sentence_list = data[0] + data[1]
        self.sentence = sentence_list
        self._model.build_vocab(sentence_list, tokenize=True)

    def predict(self, data):
        embedded_s1 = self._model.encode(data[0])
        embedded_s2 = self._model.encode(data[1])
        predictions = 1 - paired_cosine_distances(embedded_s1, embedded_s2)
        (predictions + self.min_label) * self.label_range
        return predictions

    def save(self, path):
        joblib.dump(self.sentence, path)

    def load(self, path):
        sentence_list = joblib.load(path)
        self._model.build_vocab(sentence_list, tokenize=True)


class EnsembleModel:
    """
    Class of global model
    """

    def __init__(self, models, params, name='ens_model', weights_dict=None):
        """

        :param models: list of models, models should inherit from BaseModel class
        :param name: name of the model
        :param weights_dict: weights of each model
        """
        self.name = name
        self.models = models
        self.weights = weights_dict if weights_dict else {mod.name: 1 / len(models) for mod in models}
        self.args = params

    # TODO learn weight
    def fit(self, data, y, verbose=True):
        """
        Fit all the models

        :param data: dictionary containing all types of data necessary for the models, example :
                            {
                                'reg': data_df (pandas dataframe),
                                'nn': data_iterator (torchtext iterator containing the labels)
                             }
        :param y: array-like labels,
        :param params_dict: dictionary, keys are the name of each model, values are dictionaries of training parameters
        :param verbose:
        :return:
        """
        for model in self.models:
            if verbose:
                logger.info(f'Fitting {model.name}')
            model.fit(data=data[model.dtype], y=y)

    def pred_all(self, data, verbose=True, **kwargs):
        """
        Compute the predictions of each model and the predicitons of the ensemble model

        :param data: dictionary containing all types of data necessary for the models, example :
                {
                    'reg': data_df (pandas dataframe),
                    'nn': data_iterator (torchtext iterator without shuffling, not necessarily containing the labels)
                 }
        :param verbose:
        :param kwargs: should not be necessary
        :return: pandas dataframe of predictions
        """
        predictions = pd.DataFrame()
        for model in self.models:
            if verbose:
                logger.info(f'Prediction of model {model.name}')
            predictions.loc[:, model.name] = model.predict(data[model.dtype], **kwargs)
        predictions.loc[:, self.name] = predictions.apply(lambda x: x * self.weights[x.name], axis=0).sum(1)
        return predictions

    def predict(self, data, **kwargs):
        """
        Return only the prediction of the ensemble model
        :param data: dictionary containing all types of data necessary for the models, example :
                {
                    'reg': data_df (pandas dataframe),
                    'nn': data_iterator (torchtext iterator without shuffling, not necessarily containing the labels)
                 }
        :param kwargs:
        :return: pandas series with name 'score' and index by sentence id
        """
        res = self.pred_all(data, verbose=False, **kwargs)[self.name]
        res.name = 'score'
        return res

    def eval_all(self, data, y_true=None, metrics=pearsonr_value):
        """
        Evluate the performance of each models and the performance of the ensemble model

        :param data: dictionary containing all types of data necessary for the models, example :
                {
                    'reg': data_df (pandas dataframe),
                    'nn': data_iterator (torchtext iterator without shuffling, not necessarily containing the labels)
                 }e labels
        :param y_true: array-like labels
        :param metrics: callable
        :return: pandas series
        """
        predictions = self.pred_all(data, verbose=False)
        evaluation = predictions.apply(lambda x: metrics(x, y_true))
        return evaluation

    def evaluate(self, data, y_true, metrics=pearsonr_value):
        """
        Evaluate the performance of the ensemble model
        :param data: dictionary containing all types of data necessary for the models, example :
                {
                    'reg': data_df (pandas dataframe),
                    'nn': data_iterator (torchtext iterator without shuffling, not necessarily containing the labels)
                 }
        :param y_true: array-like labels
        :param metrics: callable
        :return: pandas series
        """
        res = metrics(self.predict(data), y_true)
        return res

    def save(self, folder_path, name, extension='.pkl'):
        attributes = self.__dict__.copy()
        attributes.pop('models')
        joblib.dump(attributes, os.path.join(folder_path, f'{name}_attributes.pkl'))
        for m in self.models:
            path = os.path.join(folder_path, f'{name}_{m.name}{extension}')
            m.save(path)

    def load_model(self, folder_path, name, extension='.pkl'):
        attributes = joblib.load(os.path.join(folder_path, f'{name}_attributes.pkl'))
        for k, v in attributes.items():
            self.__dict__[k] = v
        for m in self.models:
            path = os.path.join(folder_path, f'{name}_{m.name}{extension}')
            m.load(path)

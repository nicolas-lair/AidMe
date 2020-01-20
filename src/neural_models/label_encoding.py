import torch
from torch.autograd import Variable
import numpy as np


def encode_binary_labels(labels, max_label, min_label):
    """
    Encode labels of similarity into probability distribution of similarity by simpling sclaing the label
    between 0 and 1.
    :param labels: torch Tensor
    :param max_label: maximum possible value of a label
    :param min_label: minimum possible value of a label
    :return: torch Variable of dimension (len(labels), 2)
    """
    if isinstance(labels, Variable):
        labels = labels.data

    y = labels.sub(min_label)
    y = torch.div(y, max_label - min_label)
    y = y.view(len(y), 1)
    return Variable(torch.cat([y, 1 - y], dim=1))


def decode_binary_labels(encoded_labels, max_label, min_label):
    """

    :param encoded_labels:
    :param max_label: maximum possible value of a label
    :param min_label: minimum possible value of a label
    :return:
    """
    if isinstance(encoded_labels, Variable):
        encoded_labels = encoded_labels.data

    return encoded_labels[:, 0] * (max_label - min_label) + min_label


def enc_lab(y, nclass=6):
    """
    Encode a label into a tensor as done in : Improved Semantic Representations From Tree-Structured Long Short-Term
    Memory Networks, Tai and al., 2015
    example : y = 3.7 and nclass = 6
    Y = [0., 0., 0., 0.3, 0.7, 0.]

    :param y: float representing the label, between min_sim and max_sim
    :param nclass: number of class ie max_sim - min_sim + 1
    :return: torch.Tensor of dimension (1, nclass)
    """
    Y = torch.Tensor(1, nclass).zero_()
    if y == nclass - 1:
        Y[0, nclass - 1] = 1
    else:
        Y[0, int(np.floor(y)) + 1] = y - np.floor(y)
        Y[0, int(np.floor(y))] = np.floor(y) - y + 1
    return Y


def encode_multiclass_labels(labels, nclass=6):
    """
    Encode a one-dimension iterable object containing labels using enc_lab function

    :param labels: one-dimension iterable object containing the labels
    :param nclass: Variable containing the encoded labels
    :return:
    """
    if isinstance(labels, Variable):
        labels = labels.data

    encoding = []
    for l in labels:
        encoding.append(enc_lab(l, nclass))
    return Variable(torch.cat(encoding))


def decode_multiclass_labels(encoded_labels, nclass):
    """
    Decode encoded labels

    :param encoded_labels: Array or Tensor or Variable object
    :param nclass:
    :return:
    """
    if isinstance(encoded_labels, Variable):
        encoded_labels = encoded_labels.data

    temp = torch.arange(0, nclass, 1.)
    return torch.matmul(encoded_labels, temp)

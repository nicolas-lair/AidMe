import torch
import pandas as pd
# import seaborn as sns
from scipy.stats import pearsonr

from .utils import transfer_to_gpu


def compute_predictions(model, data_iterator):
    """
    Compute the predictions given a model and a data_iterator
    :param model:
    :param data_iterator:
    :return: tuple of torch Tensor : predicted labels and gold labels on cpu
    """
    log_predictions = []
    labels = []
    gpu_compute = model.use_gpu and torch.cuda.is_available()
    if gpu_compute:
        model = model.cuda()

    for sample in data_iterator:
        x1 = sample.sent1
        x2 = sample.sent2
        try:
            labels.append(sample.score)
        except AttributeError:
            pass

        if gpu_compute:
            x1, x2 = transfer_to_gpu((x1, x2))

        log_preds = model(x1, x2)
        log_predictions.append(log_preds)
    predictions = torch.exp(torch.cat(log_predictions)).cpu()
    predicted_labels = model.decode_labels(predictions)

    try:
        labels = torch.cat(labels).data
    except RuntimeError:
        labels = None

    return predicted_labels, labels


def eval_model(model, test_iter, plot=False, verbose=False):
    """
    Compute a simple evaluation of the model given a data iterator, with computation of pearson correlation and a graph
    :param model:
    :param test_iter:
    :param plot:
    :param verbose:
    :return: pandas dataframe with two columns : 'prediction', 'label'
    """
    preds_test, test_labels = compute_predictions(model, test_iter)
    df = pd.DataFrame([preds_test.numpy(), test_labels.numpy()], index=['prediction', 'label']).T

    if verbose:
        print(f'pearson correlation : {pearsonr(df.prediction, df.label)} \n')
        print(df.head())

    if plot:
        sns.regplot(x='prediction', y='label', data=df)

    return df



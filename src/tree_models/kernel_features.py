from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cosine, cityblock, euclidean
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import sigmoid_kernel, rbf_kernel, laplacian_kernel


def linear_kernels(x, y):
    """Compute linear kernel based similarity features for two elements x and y

    Args:
        x: One row Sparse matrix representing a sentence
        y: One row Sparse matrix representing a sentence

    Returns:
        dict: kernel based similarity features
    """
    kernels = {
        'cosine': cosine,
        'manhattan': cityblock,
        'euclidean': euclidean
    }
    scores = {}

    try:
        x = x.T.toarray()
        y = y.T.toarray()
    except:
        pass

    for k, v in kernels.items():
        scores[k] = v(x, y)
    return scores


def stat_kernels(x, y):
    """Compute statistic measure based linear kernel based similarity
    features for two elements x and y

    Args:
        x: One row sparse matrix representing a sentence
        y: One row sparse matrix representing a sentence

    Returns:
        dict: kernel based similarity features
    """
    kernels = {
        'pearson': pearsonr,
        'spearman': spearmanr,
        'kendalltau': kendalltau
    }
    scores = {}

    try:
        x = x.T.toarray()
        y = y.T.toarray()
    except:
        pass

    x, y = x.reshape(-1), y.reshape(-1)
    for k, v in kernels.items():
        s = v(x, y)
        if k == 'pearson':
            scores[k] = s[0]
        else:
            scores[k] = s.correlation

    return scores


def non_linear_kernels(x, y):
    """Compute non linear kernel based similarity features for two elements x and y

    Args:
        x: One row sparse matrix representing a sentence
        y: One row sparse matrix representing a sentence

    Returns:
        dict: kernel based similarity features
    """
    kernels = {
        'sigmoid': sigmoid_kernel,
        'rbf': rbf_kernel,
        'laplacian': laplacian_kernel
    }

    scores = {}

    for k, v in kernels.items():
        scores[k] = v(x, y)[0][0]

    return scores


def compute_kernels(x, y):
    """Compute kernel based similarity features for two elements x and y

    Args:
        x: One row sparse matrix representing a sentence
        y: One row sparse matrix representing a sentence

    Returns:
        dict: kernel based similarity features
    """
    features = {}
    features.update(linear_kernels(x, y))
    features.update(stat_kernels(x, y))
    features.update(non_linear_kernels(x, y))
    # print(features)
    return features


def compute_kernels_in_batch(sent1_list, sent2_list):
    """Compute kernels for list of instances.
    The two list must have the same length and each element have exactly another
    corresponding element at the same index.

    Args:
        sent1_list: List of sentence features
        sent2_list: List of sentence features

    Returns:
        dataframe: Dataframe of similarity scores
    """
    features = []
    for i in tqdm(range(len(sent1_list))):
        features.append(compute_kernels(sent1_list[i], sent2_list[i]))

    return pd.DataFrame(features)

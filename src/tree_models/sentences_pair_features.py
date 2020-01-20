#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import pandas as pd
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from . import lemmatizer
from . import word_error_rate


# def py2_metrics(seq1, seq2):
#     """DEPRECATED: Nlg eval metrics are based on pycocoevalcap
#     And this lib is python2 dependent.

#     Computing MT based metrics using nlgeval package

#     TODO: Find an elegant way to call python2 script
#     TODO: Add alignment metrics

#     Args:
#         seq1(str): Sentence
#         seq2(str): Sentence

#     Returns:
#         metrics(dict): MT based metrics
#     """
#     current_dir = os.path.dirname(__file__)
#     output_dir = current_dir + '/py2/tmp'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     cmd = "python2 {}/py2/main.py --sentence1 '{}' --sentence2 '{}' --output_dir {}".format(current_dir, seq1, seq2,
#                                                                                             output_dir)
#     # os.popen(cmd)
#     # subprocess.Popen(shlex.split(cmd))
#     subprocess.call(shlex.split(cmd))
#     metrics = pickle.load(open('{}/temp_metric_py2.p'.format(output_dir), 'rb'), encoding='latin1')
#     return metrics


# N-gram overlaps
def ngo(list1, list2):
    """N-gram overlap as defined in Saric et al., 2012 [1]

    References:
        [1]: Šarić, F., Glavaš, G., Karan, M., Šnajder, J., & Bašić, B. D. (2012, June). 
        Takelab: Systems for measuring semantic text similarity. 
        In Proceedings of the First Joint Conference on Lexical and Computational Semantics-Volume 1: 
        Proceedings of the main conference and the shared task, and 
        Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (pp. 441-448). 
        Association for Computational Linguistics.
    
    Args:
        list1(list): First n_gram
        list2(list): Second n_gram
    
    Returns:
        float: N-gram overlap coefficient
    """
    intersection = list(set(list1).intersection(list2))
    n_inter = len(intersection)
    n_list1 = len(list1)
    n_list2 = len(list2)
    if n_inter == 0:
        return 0
    result = 2 / (n_list1 / n_inter + n_list2 / n_inter)
    return result


def ngo_with_grams(seq1, seq2, n, level='c'):
    """N-gram overlap given sentences

    Args:
        seq1: Sentence from which to compute n-grams
        seq2: Sentence from which to compute n-grams
        n: Order of the n-gram sequences
        level (str): Can be 'l' for lemmas and 'w' for word level
    
    Returns:
        tuple: n-grams for each sequence and the relevant n-gram overlap metric
    """
    if level == 'c':
        pass

    if level == 'w':
        seq1 = lemmatizer.tokenize(seq1)
        seq2 = lemmatizer.tokenize(seq2)

    if level == 'l':
        seq1 = lemmatizer.lemmatize(seq1)
        seq2 = lemmatizer.lemmatize(seq2)

    grams_1 = list(nltk.ngrams(seq1, n))
    grams_2 = list(nltk.ngrams(seq2, n))
    return ngo(grams_1, grams_2)


# Sequence Features
def levenshtein_distance(sentence1, sentence2):
    """Compute Levenshtein distance i.e minimum numer of editing operations needed to convert one string 
    into another. The editing operations can consist of insertions, deletions and substitutions.
    This is just edit_distance of nltk package.
    As preprocessing step, we remove stop words then lemmatize the sentence as done in [1]
    
    References:
        [1]: Tian, J., Zhou, Z., Lan, M., & Wu, Y. (2017). ECNU at SemEval-2017 Task 1: 
            Leverage Kernel-based Traditional NLP features and Neural Networks to Build a 
            Universal Model for Multilingual and Cross-lingual Semantic Textual Similarity. 
            In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017) 
            (pp. 191-197).
    Args:
        sentence1(str): Sentence 
        sentence2(str): Sentence
    
    Returns:
        float: Levenshtein distance
    """
    seq1 = lemmatizer.lemmatize(sentence1, ignore_stop_words=True)
    seq2 = lemmatizer.lemmatize(sentence2, ignore_stop_words=True)
    return nltk.edit_distance(seq1, seq2, transpositions=False) / max(len(seq1), len(seq2))


# MT based metrics
def nist_score(sentence1, sentence2):
    """TODO

    Args:
        sentence1(str): Sentence 
        sentence2(str): Sentence
    
    Returns:
        float: NIST Score
    """
    pass


def bleu_score(sentence1, sentence2, n=4):
    """Compute bleu scores

    Args:
        sentence1(str): Sentence 
        sentence2(str): Sentence
        n(int): N-gram order
    
    Returns:
        list: Bleu scores
    """
    return Bleu(n).compute_score({0: [sentence1]}, {0: [sentence2]})[0]


def meteor_score(sentence1, sentence2):
    """Compute METEOR score
    
    Args:
        sentence1(str): Sentence 
        sentence2(str): Sentence
    
    Returns:
        float: METEOR scores
    """
    return Meteor().compute_score({0: [sentence1]}, {0: [sentence2]})[0]


def rouge_score(sentence1, sentence2):
    """Compute ROUGE score
    
    Args:
        sentence1(str): Sentence 
        sentence2(str): Sentence
    
    Returns:
        float: ROUGE score
    """
    return Rouge().compute_score({0: [sentence1]}, {0: [sentence2]})[0]


def wer_score(sentence1, sentence2):
    """Same as levenshtein distance in case of no translation

    Args:
        sentence1(str): Element of the pair of sentences
        sentence2(str): Element of the pair of sentences
    
    Returns:
        float: 1-Word Error Rate
    """
    seq1 = lemmatizer.tokenize(sentence1)
    seq2 = lemmatizer.tokenize(sentence2)
    return 1 - word_error_rate.wer(seq1, seq2)


def mt_overlap_scores(sentence1, sentence2, ignore_meteor=False):
    """Compute machine translation (MT) overlap scores (Bleu, METEOR and ROUGE_L) given two sentences
    The option 'ignore_meteor' is a boolean set to true in case we should not compute METEOR due to computation 
    time for example.
    
    Args:
        sentence1(str): Reference string
        sentence2(str): Hypothesis string 
        ignore_meteor(bool): Set to true if METEOR is not computed

    Returns:
        dict: Dictionary containing the name of the MT metrics
    """
    scorers = [
        (bleu_score, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (meteor_score, "METEOR"),
        (rouge_score, "ROUGE_L")
    ]
    if ignore_meteor:
        scorers = [
            (bleu_score, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (rouge_score, "ROUGE_L")
        ]
    scores = {}
    for scorer, method in scorers:
        res = scorer(sentence1, sentence2)
        if type(method) == list:
            for k in range(len(method)):
                scores[method[k]] = res[k]
        else:
            scores[method] = res
    return scores


def mt_overlap_scores_list(sentence1, sentence2, ignore_meteor=False):
    """Compute machine translation (MT) overlap scores (Bleu, METEOR and ROUGE_L) given two sentences
    The option 'ignore_meteor' is a boolean set to true in case we should not compute METEOR due to computation 
    time for example.
    Stored in a list to keep order the same.

    Args:
        sentence1(str): Reference string
        sentence2(str): Hypothesis string 
        ignore_meteor(bool): Set to true if METEOR is not computed

    Returns:
        list: List of tuples
    """
    scorers = [
        (bleu_score, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (meteor_score, "METEOR"),
        (rouge_score, "ROUGE_L")
    ]
    if ignore_meteor:
        scorers = [
            (bleu_score, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (rouge_score, "ROUGE_L")
        ]
    scores = []
    for scorer, method in scorers:
        res = scorer(sentence1, sentence2)
        if isinstance(res, list):
            for k in range(len(res)):
                scores.append((method[k], res[k]))
        else:
            scores.append((method, res))
    return scores


def sentence_pair_features(sentence1, sentence2, ignore_meteor=True):
    """Compute all traditional features according to the configuration 
    specified in [1] i.e n-grams overlaps (word, lemma and char levels each with 3 or 4 orders),
    sequence features (levenshetien distance), mt based features(bleu, meteor, rouge).

    References:
        [1]: Tian, J., Zhou, Z., Lan, M., & Wu, Y. (2017). 
            ECNU at SemEval-2017 Task 1: Leverage Kernel-based Traditional NLP features and Neural Networks to Build a 
            Universal Model for Multilingual and Cross-lingual Semantic Textual Similarity. 
            In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017) (pp. 191-197).

    Args:
        sentence1(str): Reference sentence
        sentence2(str): Hypothesis sentence
        ignore_meteor(bool): Set to true if METEOR is not computed
           
    Returns:
        list: List containing features
    """
    # modes = ['c', 'w', 'l']
    n_char_level = [2, 3, 4, 5]
    n_word_level = [1, 2, 3]
    features = []

    for n in n_char_level:
        features.append(ngo_with_grams(sentence1, sentence2, n, level='c'))
    for n in n_word_level:
        features.append(ngo_with_grams(sentence1, sentence2, n, level='w'))
    for n in n_word_level:
        features.append(ngo_with_grams(sentence1, sentence2, n, level='l'))

    features.append(levenshtein_distance(sentence1, sentence2))
    features.append(wer_score(sentence1, sentence2))
    metrics_list = mt_overlap_scores_list(sentence1, sentence2, ignore_meteor=ignore_meteor)
    for _, val in metrics_list:
        features.append(val)
    return features


def compute_sentence_pair_features(sent1_batch, sent2_batch, ignore_meteor=True):
    """Compute all traditional features according to the configuration for a batch of samples
    specified in [1] i.e n-grams overlaps (word, lemma and char levels each with 3 or 4 orders),
    sequence features (levenshetien distance), mt based features(bleu, meteor, rouge).

    References:
        [1]: Tian, J., Zhou, Z., Lan, M., & Wu, Y. (2017). 
            ECNU at SemEval-2017 Task 1: Leverage Kernel-based Traditional NLP features and Neural Networks to Build a 
            Universal Model for Multilingual and Cross-lingual Semantic Textual Similarity. 
            In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017) (pp. 191-197).

    Args:
        sent1_batch(str): List of Reference sentence
        sent2_batch(str): List of Hypothesis sentence
        ignore_meteor(bool): Set to true if METEOR is not computed
           
    Returns:
        DataFrame: List containing features
    """
    features = []
    for i in tqdm(range(len(sent1_batch))):
        features.append(sentence_pair_features(sent1_batch[i], sent2_batch[i], ignore_meteor))
    features_names = [
        'Ngo-char-2', 'Ngo-char-3', 'Ngo-char-4', 'Ngo-char-5',
        'Ngo-word-1', 'Ngo-word-2', 'Ngo-word-3',
        'Ngo-lemma-1', 'Ngo-lemma-2', 'Ngo-lemma-3',
        'lev_distance', 'WER',
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L'
    ]
    features = pd.DataFrame(features)
    features.columns = features_names
    return pd.DataFrame(features)

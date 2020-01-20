import os

import joblib
import numpy as np

import loader
import logger as lg
from tree_models import bag_of_words
from tree_models import kernel_features, word_embedding_transformer
from tree_models.sentences_pair_features import compute_sentence_pair_features

logger = lg.logging.getLogger(name='preprocessing')
logger.setLevel(lg.logging.DEBUG)

DATA_FOLDER = '/home/nicolas/PycharmProjects/similarity-measures/data'
DATASET = 'semeval'
NAME= 'semeval_train_dev'
TRAIN_OR_TEST = 'test'

if DATASET == 'semeval':
    file_name = f'clean_test.csv'
else:
    raise NotImplementedError

args = {
    'output_folder': os.path.join('/home/nicolas/PycharmProjects/similarity-measures/src/output', NAME),
    'DATASET': DATASET,
    'tfidf_file': 'tf_idf.pkl',
    'embedding_folder': os.path.join(DATA_FOLDER, 'word_embedding'),
    'embedding_list': ['paragram_300_sl999.txt', 'glove.6B.300d.txt'],
    'train_features': True if TRAIN_OR_TEST == 'train' else False,
    'compute_tfidf': True if TRAIN_OR_TEST == 'train' else False
}

if DATASET == 'semeval':
    args['data_path'] = os.path.join('/home/nicolas/PycharmProjects/similarity-measures/src/output/semeval', file_name)
    args['file_headers'] = ['sent1', 'sent2', 'score']
    args['use_pair_features'] = True
    args['use_wef'] = True
    args['use_bow_features'] = True
    args['df_loader'] = loader.csv_to_df
    args['dts_loader'] = loader.csv_to_dataset


def compute_bow_features(df, tfidf_bow):
    # Computing bag of words of features
    sent1_bow = list(tfidf_bow.transform(list(df.sent1)))
    sent2_bow = list(tfidf_bow.transform(list(df.sent2)))

    # # Applying kernels
    bow_features_df = kernel_features.compute_kernels_in_batch(sent1_bow, sent2_bow)
    bow_features_df.columns = [
        name + '_bow' for name in bow_features_df.columns]
    return bow_features_df


def compute_wef_features(tokenized_data, embedding_folder, embedding_list, idf_dict, corpus_size):
    """
    Computing Word Embedding Features
    Loading DATASET and tokenizing each sample
    :param tokenized_data:
    :param embedding_folder:
    :param embedding_list:
    :param idf_dict:
    :param corpus_size:
    :return:
    """

    sent1_tokenized, sent2_tokenized = tokenized_data.sent1, tokenized_data.sent2

    word_embedder = word_embedding_transformer.WordEmbeddingTransformer(
        tokenized_data, embedding_folder, embedding_list, idf_dict, corpus_size)
    sent1_word_emb = word_embedder.transform(sent1_tokenized)
    sent2_word_emb = word_embedder.transform(sent2_tokenized)
    word_embeddings_features = kernel_features.compute_kernels_in_batch(
        sent1_word_emb, sent2_word_emb)
    word_embeddings_features.columns = [
        name + '_wef' for name in word_embeddings_features.columns]
    return word_embeddings_features


def check_nan_in_features(features_df, ind=None):
    check_nan = np.isnan(features_df).any(axis=1)
    if check_nan.any():
        idx = features_df[check_nan].index
        idx = idx if ind is None else [ind[i] for i in idx]
        logger.debug(f'Got NaN for these index{idx}')
        features_df.fillna(0, inplace=True)


def compute_features(data_path, file_headers, compute_tfidf, use_pair_features, use_bow_features, use_wef, df_loader,
                     dts_loader, output_folder, embedding_folder, embedding_list, tf_idf=None, sentence_field=None,
                     ind=None,
                     save=True, **kwargs):
    """
    Generic function to compute the features necessary for the regression model (xgboost and random forest)
    Data are loaded with specific data loader functions available in the loader module.
    :param file_headers:
    :param embedding_folder:
    :param embedding_list:
    :param data: object or path to the data, should be compatible with data loaders
    :param compute_tfidf: boolean, specify if the data are train_feature (True) or test_features (False)
    :param use_pair_features: boolean
    :param use_bow_features: boolean
    :param use_wef: boolean
    :param output_folder: path
    :param sentence_field:
    :param df_loader: load data_link as a pandas dataframe, see loader module
    :param dts_loader: load data_link as a torchtext DATASET, see loader module
    :param ind: list of int, keep only data with their index in ind
    :return: dataframe of features, score as an array if available in the data
    """
    # Create output folder if it does not exist
    if save:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    pair_features, bow_features, wef_features = None, None, None
    tfidf_bow, idf_dict = None, None

    # Load data
    df = df_loader(data_path, file_headers, ind)

    if use_pair_features:
        # logger.info('Compute sentence pair features')
        pair_features = compute_sentence_pair_features(df.sent1.values, df.sent2.values)
        check_nan_in_features(pair_features, ind)
        if save:
            pair_features.to_csv(os.path.join(args['output_folder'], f'{TRAIN_OR_TEST}_pair_features.tsv'), sep='\t',
                                 index=False)

    if use_bow_features or use_wef:
        if compute_tfidf:
            logger.info('Computing tf_idf')
            tfidf_bow, idf_dict = bag_of_words.train_tfidfbow(df, output_folder=output_folder, save=save)
        elif tf_idf is None:
            logger.info('Loading tf idf')
            tfidf_bow, idf_dict = loader.load_tfidfbow(output_folder)
        else:
            tfidf_bow, idf_dict = tf_idf

    if use_bow_features:
        # logger.info('Compute BoW features')
        bow_features = compute_bow_features(df, tfidf_bow)
        check_nan_in_features(bow_features, ind)
        if save:
            bow_features.to_csv(os.path.join(args['output_folder'], f'{TRAIN_OR_TEST}_bow_features.tsv'), sep='\t',
                                index=False)

    # logger.info('Compute Word Embedding features')
    dataset = dts_loader(data_path, headers=file_headers, keep_ind=ind, sentence_field=sentence_field)
    # Compute and save sentence_field if needed
    if sentence_field is None:
        logger.info('Computing sentence field')
        dataset.fields['sent1'].build_vocab(dataset)
        sentence_field = dataset.fields['sent1']
        if save:
            logger.info('Saving sentence_field')
            joblib.dump(sentence_field, os.path.join(output_folder, 'sentence_field.pkl'))

    if use_wef:
        wef_features = compute_wef_features(dataset, embedding_folder, embedding_list, idf_dict, tfidf_bow.corpus_size)
        check_nan_in_features(wef_features, ind)
        if save:
            wef_features.to_csv(os.path.join(output_folder, f'{TRAIN_OR_TEST}_wef.tsv'), sep='\t', index=False)

    # logger.info('Finished')
    # Return the score if the score is included in the data
    score = df.score.values if 'score' in file_headers else None
    if save:
        joblib.dump(score, os.path.join(output_folder, f'{TRAIN_OR_TEST}_scores.pkl'))
    return pair_features, bow_features, wef_features, score, sentence_field, (tfidf_bow, idf_dict)


if __name__ == "__main__":
    sentence_field = None

    try:
        logger.info('Loading sentence_field')
        sentence_field = joblib.load(os.path.join(args['output_folder'], 'sentence_field.pkl'))
    except:
        logger.debug('No sentence field could be found')
        sentence_field = None

    compute_features(sentence_field=sentence_field, **args)

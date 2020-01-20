import os
from pathlib import Path

import joblib
import torch
from torch import nn
from torchtext import vocab

import logger as lg
from UserGrammar.corpus import _generate_corpus
from UserGrammar.corpus_data import default_train_corpus
from dialog_flow import DialogFlowAgent
from oracle_agent import Agent as oracle_Agent, get_metrics

logger = lg.logging.getLogger(name='run_simulation')
logger.setLevel(lg.logging.DEBUG)

HOME = str(Path.home())
PROJECT_PATH = os.path.join(os.getcwd(), '../')
USE_GPU = False
NAME = f'simultest_600'
DIALOGFLOW = True
instance = 1 # , 1
SAVE_FOLDER = os.path.join(PROJECT_PATH, 'src', 'output', 'simulation')

# Agent=DialogFlowAgent
Agent = oracle_Agent

label_range = (0, 1)

train_params = {
    'embedding_folder': os.path.join(PROJECT_PATH, 'data', 'word_embedding'),
    'embedding_list': ['paragram_300_sl999.txt', 'glove.6B.300d.txt'],
    'embedding': 'paragram_300_sl999.txt',
    'use_pair_features': True,
    'use_wef': False,
    'use_bow_features': True,
    'max_nn_corpus': 10000,
    'model_params': {
        'xgboost': {'max_depth': 4},
        'random_forest': {'max_depth': 8},
        'binary_nn': {'name': 'binary_nn',
                      'emb_dim': 300,
                      'output_size': (32, 64, 32, 2),
                      'label_range': label_range,
                      'hidden_activation_layer': nn.ReLU(),
                      'optimizer': torch.optim.Adam,
                      'loss_func': nn.KLDivLoss(reduction="batchmean"),
                      'epochs': 5,
                      'verbose': False},
        'multi_class_nn': {'name': 'multi_class_nn',
                           'emb_dim': 300,
                           'label_range': label_range,
                           'output_size': (64, 32, 5),
                           'hidden_activation_layer': nn.ReLU(),
                           'optimizer': torch.optim.Adam,
                           'loss_func': nn.KLDivLoss(reduction="batchmean"),
                           'epochs': 5,
                           'verbose': False}

    }
}


def run(user_corpus, agent, dialogflow_agent=None, filename=None):
    metrics = []
    first_sentence_dict = user_corpus.pop(0)
    logger.info(f'Sentence number 1: {first_sentence_dict["sentence"]}')
    if agent:
        agent.update_agent(first_sentence_dict["sentence"], first_sentence_dict["intent"],
                           first_sentence_dict["pattern"], first_sentence_dict["args"],
                           new_pairs=[[first_sentence_dict["sentence"], first_sentence_dict["sentence"]]],
                           new_features=agent.compute_features(
                               pairs=[[first_sentence_dict["sentence"], first_sentence_dict["sentence"]]],
                               headers=['sent1', 'sent2']))
    if dialogflow_agent:
        dialogflow_agent.update_agent(first_sentence_dict["intent"], first_sentence_dict["pattern"],
                                      first_sentence_dict["args"])
    metrics.append(get_metrics(sentence_dict=first_sentence_dict, intent_detected=False, closest_sentence="",
                               intent_similarity_scores=(0., 0.), pattern_detected=False, attempts=- 1, new_intent=True,
                               new_pattern=True, argument_similarity_scores=[], closest_pattern=[],
                               dialogflow_duration=0, dialogflow_compare=True, dialogflow_pattern=False,
                               dialogflow_intent=False))
    agent_metrics = dict()
    dialogflow_metrics = dict()
    i = 0
    for sentence_dict in user_corpus:
        logger.info(f'Sentence number {i}: {sentence_dict["sentence"]}')
        if agent:
            agent_metrics = agent.run(sentence_dict)
        if dialogflow_agent:
            dialogflow_metrics = dialogflow_agent.run(sentence_dict)
        metrics.append({**agent_metrics, **dialogflow_metrics})
        i += 1
        if i % 25 == 0:
            joblib.dump(metrics, os.path.join(SAVE_FOLDER, f'{filename}_metrics.pkl'))

    joblib.dump(metrics, os.path.join(SAVE_FOLDER, f'{filename}_metrics.pkl'))


if __name__ == '__main__':
    logger.info('Generate corpus')
    # train_data = generate_train_corpus(corpus_size=20, n_sentence=5)
    # train_corpus, train_patterns = train_data['corpus'], train_data['used_patterns']
    # user_corpus = generate_user_corpus(number_of_sentences=30,
    #                                    train_used_patterns=train_patterns,
    #                                    proportion_of_novelty=0.25)

    #
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # user_corpus = joblib.load('/home/nicolas/Desktop/deathstar simulation/user_corpus_0')
    # vectors = vocab.Vectors(train_params['embedding'], train_params['embedding_folder'])
    # for freq in [1, 3, 5, 10]:
    #     logger.info(f'Train frequences: {freq}')
    #     agent = Agent(name=NAME + f'_deathstar_{str(freq)}', train_params=train_params, train_freq=freq, max_guess=3,
    #               use_gpu=USE_GPU, initial_train_corpus=default_train_corpus, vectors=vectors)
    #     dialogflow_agent = DialogFlowAgent(instance=instance, name=f'dialogflow_agent_{instance}', train_freq=freq)
    #     logger.info('Run simulation')
    #     run(user_corpus=user_corpus, agent=agent, dialogflow_agent=dialogflow_agent, filename=agent.name)

    vectors = vocab.Vectors(train_params['embedding'], train_params['embedding_folder'])
    train_corpus = default_train_corpus
    user_corpus = _generate_corpus(700)
    joblib.dump(user_corpus, os.path.join(SAVE_FOLDER, f'user_corpus_last'))
    train_freq_list = [1, 3, 5, 10, 25, 50]
    for freq in train_freq_list:
        logger.info(f'Train frequences: {freq}')
        agent = Agent(name=NAME + f'_last_' + str(freq), train_params=train_params, train_freq=freq, max_guess=3,
                      use_gpu=USE_GPU, initial_train_corpus=train_corpus, vectors=vectors)
        dialogflow_agent = DialogFlowAgent(instance=instance, name=f'dialogflow_agent_{instance}', train_freq=freq)

        logger.info('Run simulation')
        run(user_corpus=user_corpus, agent=agent, dialogflow_agent=dialogflow_agent, filename=agent.name)

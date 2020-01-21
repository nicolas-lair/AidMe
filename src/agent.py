import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from torchtext.data import Field, get_tokenizer

import loader
import logger as lg
from UserGrammar.corpus_data import default_train_corpus
from UserGrammar.generator import generate
from tree_models.bag_of_words import TfidfBagOfWords
from model import EnsembleModel
from model import NeuralModel
from preprocessing import compute_features as comp_feat
from utils import prepare_data_nn, create_models_list

logger = lg.logging.getLogger(name='agent')
logger.setLevel(lg.logging.DEBUG)


def validate(predictions, answers):
    """
    Check if the prediction is in the answers (intent or full sentence)
    :param predictions: string
    :param answers: list of string
    :return: boolean
    """
    return np.array([p in answers for p in predictions])


# TODO add auto_eval function


def get_metrics(sentence_dict, intent_detected, closest_sentence, intent_similarity_scores, new_intent,
                new_pattern, pattern_detected=False, attempts=-1, closest_pattern=(), argument_similarity_scores=(),
                intent_duration=0, pattern_duration=0, dialogflow_compare=False, dialogflow_duration=None,
                dialogflow_intent=None, dialogflow_pattern=None):
    metric = {
        'sentence': sentence_dict['sentence'],  # string
        'number_of_variables': sentence_dict["number_of_variables"],
        'number_of_domains': sentence_dict["number_of_domains"],
        'complexity': sentence_dict["complexity"],
        'new_intent': new_intent,
        'new_pattern': new_pattern,

        # duration
        'intent_duration': intent_duration,
        'pattern_duration': pattern_duration,
        'dialogflow_duration': dialogflow_duration,

        # Intent prediction metrics
        'intent_detected': intent_detected,  # bool
        'closest_sentence': closest_sentence,  # string
        'similarity_scores_by_intent': intent_similarity_scores[0],  # float
        'similarity_scores_by_pattern': intent_similarity_scores[1],  # float

        # Argument matching metrics
        'pattern_detected': pattern_detected,  # bool
        'closest_patterns': closest_pattern,  # list of string
        'attempts': attempts,  # int
        'argument_similarity_scores': argument_similarity_scores,  # list of float
    }

    if dialogflow_compare:
        # duration
        metric['dialogflow_duration']: dialogflow_duration

        # dialogflow
        metric['dialogflow_intent']: dialogflow_intent
        metric['dialogflow_pattern']: dialogflow_pattern

    return metric


class Agent:
    def __init__(self, name, train_params, train_freq, max_guess, vectors, initial_train_corpus=default_train_corpus,
                 use_gpu=True):
        self.name = name
        self.initial_train_corpus = initial_train_corpus
        self.model = EnsembleModel([], train_params)

        # params
        self.train_params = train_params
        self.train_freq = train_freq
        self.since_last_train = 0
        self.max_guess = max_guess
        self.use_gpu = use_gpu

        # memory
        self.construction_memory = {}
        self.sentence_lookup_table = OrderedDict()

        # train data, features
        # self.vectors = vocab.Vectors(self.train_params['embedding'], self.train_params['embedding_folder'])
        self.vectors = vectors
        self.corpus = self.initial_train_corpus

        initial_sentence_list = [c[0] for c in self.corpus] + [c[1] for c in self.corpus]
        self.tf_idf = self.update_tfidf(data=initial_sentence_list)
        self.sentence_field = Field(sequential=True, tokenize=get_tokenizer('spacy'), lower=True)
        self.update_vocab(data=initial_sentence_list)
        self.features_df = self.compute_features(self.corpus, headers=['sent1', 'sent2', 'score'])

        models_list = create_models_list(self.train_params['model_params'], self.sentence_field, self.use_gpu)
        self.model = EnsembleModel(models_list, self.train_params)
        self.update_model()
        # self.intent_confidence = 0.8
        # self.sentence_confidence = 0.9

    def update_tfidf(self, data=None):
        data = list(self.sentence_lookup_table.keys()) if data is None else data
        tf = TfidfBagOfWords(lowercase=False)
        tf.fit(data)
        self.tf_idf = tf, tf.get_idf_dict()
        return self.tf_idf

    def update_vocab(self, data=None):
        data = list(self.sentence_lookup_table.keys()) if data is None else data
        dts = loader.list_to_dataset([[d] for d in data], headers=['sent1'], sentence_field=self.sentence_field)
        self.sentence_field.build_vocab(dts, vectors=self.vectors)

        for m in self.model.models:
            if isinstance(m, NeuralModel):
                m.update_embedding_layer(self.sentence_field.vocab)
        return self.sentence_field

    def update_agent(self, sentence, intent, pattern, args, new_pairs, new_features):
        self.update_construction_memory(intent, sentence, pattern, args)
        self.update_lookup_table(sentence, intent, pattern)
        self.update_corpus(new_pairs)
        self.update_features_df(new_features)

    def update_lookup_table(self, sentence, intent, pattern):
        self.sentence_lookup_table[sentence] = {'intent': intent, 'pattern': pattern}

    def update_construction_memory(self, intent, sentence, pattern, args):
        if intent in self.construction_memory.keys():
            self.construction_memory[intent]['patterns'].add(pattern)
            self.construction_memory[intent]['sentences'].add(sentence)
            for k, v in args.items():
                self.construction_memory[intent][k].add(v)
        else:
            args = {k: {v} for k, v in args.items()}
            self.construction_memory.update({intent: {'patterns': {pattern}, 'sentences': {sentence}, **args}})

    def update_corpus(self, new_pairs):
        scores = [self.sentence_lookup_table[s1]['intent'] == self.sentence_lookup_table[s2]['intent'] for s1, s2 in
                  new_pairs]
        new_scored_pairs = [[*pair, sc] for pair, sc in zip(new_pairs, scores)]
        self.corpus += new_scored_pairs

    def update_features_df(self, features):
        try:
            self.features_df = pd.concat([self.features_df, features])
        except ValueError:
            pass

    def update_model(self):
        train_data, scores = self.build_train_corpus()

        logger.info('Creating models')
        models_list = create_models_list(self.train_params['model_params'], self.sentence_field, self.use_gpu)
        self.model = EnsembleModel(models_list, self.train_params)

        logger.info('Training')
        self.model.fit(train_data, scores)

    def build_train_corpus(self):
        if len(self.corpus) > self.train_params['max_nn_corpus']:
            nn_corpus = [self.corpus[idx] for idx in
                         np.random.choice(len(self.corpus), self.train_params['max_nn_corpus'], replace=False)]
        else:
            nn_corpus = self.corpus
        data_nn = prepare_data_nn(nn_corpus, headers=['sent1', 'sent2', 'score'], sentence_field=self.sentence_field,
                                  dts_loader=loader.list_to_dataset, train=True, **self.train_params)

        formatted_data = {
            'reg': self.features_df,
            'nn': data_nn
        }
        scores = np.array([c[2] for c in self.corpus])
        assert len(scores) == len(self.features_df)
        return formatted_data, scores

    def compute_features(self, pairs, headers):
        pair_features, bow_features, wef_features, _, _, _ = comp_feat(
            pairs,
            file_headers=headers,
            compute_tfidf=False,
            tf_idf=self.tf_idf,
            sentence_field=self.sentence_field,
            df_loader=loader.list_to_df,
            dts_loader=loader.list_to_dataset,
            save=False,
            output_folder=None,  # unused, just to ensure compatibility
            **self.train_params)
        try:
            features = pd.concat([pair_features, bow_features, wef_features], axis=1)
        except ValueError:
            features = None

        return features

    def compare_sentence_to_list(self, sentence, sentence_list):
        pairs = [[sentence, s] for s in sentence_list]
        logger.debug(f'Formatting {len(pairs)} pairs for prediction')
        features = self.compute_features(pairs=pairs, headers=['sent1', 'sent2'])
        data_nn = prepare_data_nn(pairs, headers=['sent1', 'sent2'], sentence_field=self.sentence_field,
                                  dts_loader=loader.list_to_dataset, train=False, **self.train_params)
        formatted_data = {
            'reg': features,
            'nn': data_nn
        }

        # logger.debug('Begin score_computation')
        scores = self.model.predict(formatted_data)
        return pairs, features, scores

    def predict_intent(self, sentence):  # TODO score threshold (learn vs imposed)
        # logger.debug('Compute similarity score between the new sentence and the known sentences')
        known_sentences = list(self.sentence_lookup_table.keys())
        sentence_pairs, formatted_data, scores = self.compare_sentence_to_list(sentence, known_sentences)
        scores = pd.DataFrame(scores).head(-1)  # Remove the pair (sentence, sentence)

        # logger.debug('Identify closest intent and pattern')
        # Obscure but gets patterns and intents list from list l of tuples : list(zip(*l))
        intents, patterns = list(zip(*[
            (s['intent'], s['pattern']) for s in
            list(self.sentence_lookup_table.values())[:-1]]))  # Remove the pair (sentence, sentence)
        # intents, patterns = list(zip(*[self.sentence_lookup_table[s] for s in known_sentences[:-1]]))
        scores['patterns'] = list(patterns)
        scores['intents'] = list(intents)

        scores_by_intents = scores.groupby('intents').mean().sort_values(by='score', ascending=False).head(1)
        closest_intent = scores_by_intents.index[0]
        intent_similarity_score = scores_by_intents.loc[closest_intent, 'score']
        scores = scores[scores['intents'] == closest_intent]

        scores_by_patterns = scores.groupby('patterns').mean().sort_values(by='score', ascending=False).head(1)
        closest_pattern = scores_by_patterns.index[0]
        pattern_similarity_score = scores_by_patterns.loc[closest_pattern, 'score']

        closest_sentence = known_sentences[scores.loc[scores['patterns'] == closest_pattern].index[0]]

        return closest_intent, closest_pattern, closest_sentence, (
            intent_similarity_score, pattern_similarity_score), (sentence_pairs, formatted_data)

    def match_argument(self, sentence, closest_pattern):
        # logger.debug('Generate candidate patterns')
        candidate_sentences_dict = generate(closest_pattern, sentence)
        candidate_sentences = list(candidate_sentences_dict.keys())
        candidate_patterns = list(candidate_sentences_dict.values())

        # logger.debug('Get similarity scores')
        _, _, scores = self.compare_sentence_to_list(sentence, candidate_sentences)

        # logger.debug('Identify best patterns')
        scores = scores.sort_values(ascending=False).head(self.max_guess)
        best_index = scores.index
        best_sentences = [candidate_sentences[i] for i in best_index]
        best_patterns = [candidate_patterns[i] for i in best_index]
        return best_sentences, best_patterns, scores.values

    def is_new_intent_and_pattern(self, intent, pattern):
        if intent in self.construction_memory.keys():
            is_new_intent = False
            if pattern in self.construction_memory[intent]['patterns']:
                logger.info('Intent and pattern are known')
                is_new_pattern = False
            else:
                logger.info('This is a new pattern but intent is known')
                is_new_pattern = True
        else:
            logger.info('This is new intent (and new pattern)')
            is_new_intent = True
            is_new_pattern = True

        return is_new_intent, is_new_pattern

    def run(self, sentence_dict):
        # first_sentence_dict = user_corpus.pop(0)
        # logger.info(f'Sentence number 1: {first_sentence_dict["sentence"]}')
        # self.update_agent(first_sentence_dict["sentence"], first_sentence_dict["intent"],
        #                   first_sentence_dict["pattern"], first_sentence_dict["args"],
        #                   new_pairs=[[first_sentence_dict["sentence"], first_sentence_dict["sentence"]]],
        #                   new_features=self.compute_features(
        #                       pairs=[[first_sentence_dict["sentence"], first_sentence_dict["sentence"]]],
        #                       headers=['sent1', 'sent2']))
        # self.metrics.append(
        #     get_metrics(sentence_dict=first_sentence_dict, intent_detected=False, closest_sentence="",
        #                      intent_similarity_scores=(0., 0.), pattern_detected=False, attempts=- 1, new_intent=True,
        #                      new_pattern=True, argument_similarity_scores=[], closest_pattern=[], ))
        #
        # i = 1
        # for sentence_dict in user_corpus:
        if self.since_last_train % self.train_freq == 0:
            self.update_model()

        true_intent = sentence_dict["intent"]
        true_pattern = sentence_dict["pattern"]
        true_args = sentence_dict["args"]
        sentence = sentence_dict["sentence"]
        number_of_var = sentence_dict["number_of_variables"],
        possible_patterns = sentence_dict["possible_patterns"]
        possible_sentences = sentence_dict["possible_sentences"]

        is_new_intent, is_new_pattern = self.is_new_intent_and_pattern(true_intent, true_pattern)

        t = time.time()

        if sentence in self.sentence_lookup_table.keys():
            logger.info('This exact sentences was known before')
            sentence_metrics = get_metrics(sentence_dict=sentence_dict, intent_detected=True,
                                           closest_sentence=sentence, intent_similarity_scores=(1., 1.),
                                           pattern_detected=True, new_intent=is_new_intent,
                                           new_pattern=is_new_pattern)
        else:
            self.sentence_lookup_table[sentence] = {}
            self.update_vocab()
            self.update_tfidf()

            closest_intent, closest_pattern, closest_sentence, intent_similarity_scores, (
                new_pairs, new_features) = self.predict_intent(sentence)
            intent_duration = round(time.time() - t, 3)
            intent_detected = (closest_intent == true_intent)
            if intent_detected:
                logger.info(f'Intent detection OK:  {closest_pattern}')
                if number_of_var == 0:
                    pattern_detected = True
                    attempt_success = -1
                    argument_similarity_scores = []
                    pattern_duration = 0
                else:
                    t = time.time()
                    predicted_sentences, predicted_patterns, argument_similarity_scores = \
                        self.match_argument(sentence, closest_pattern)
                    pattern_duration = round(time.time() - intent_duration, 3)
                    success = validate(predicted_sentences, possible_sentences)
                    pattern_detected = success.any()
                    if pattern_detected:
                        attempt_success = np.flatnonzero(success)[0]
                        # assert args[attempt_success] == true_args
                        try:
                            assert predicted_patterns[attempt_success] == true_pattern
                        except AssertionError:
                            logger.error(f'Inconsistency between sentence and pattern comparison : '
                                         f'predicted: {predicted_patterns[attempt_success]}'
                                         f'true: {true_pattern}')
                        logger.info(f'Argument detection OK on the {attempt_success + 1} attempt: {true_pattern}')
                    else:
                        try:
                            assert not validate(predicted_patterns, possible_patterns).any()
                        except AssertionError:
                            logger.error(f'Inconsistency between sentence and pattern comparison : '
                                         f'predicted: {predicted_patterns}'
                                         f'possible: {possible_patterns}')
                        logger.info(f'Argument detection FAILED: {predicted_patterns}')
                        attempt_success = -1
            else:
                logger.info(f'Intent detection FAILED: {closest_pattern}')
                pattern_detected = False
                attempt_success = -1
                argument_similarity_scores = []
                closest_pattern = []
                pattern_duration = None
            sentence_metrics = get_metrics(sentence_dict=sentence_dict,
                                           intent_detected=intent_detected,
                                           closest_sentence=closest_sentence,
                                           intent_similarity_scores=intent_similarity_scores,
                                           pattern_detected=pattern_detected,
                                           attempts=attempt_success + 1,
                                           new_intent=is_new_intent,
                                           new_pattern=is_new_pattern,
                                           argument_similarity_scores=argument_similarity_scores,
                                           closest_pattern=closest_pattern,
                                           intent_duration=intent_duration,
                                           pattern_duration=pattern_duration)

            self.update_agent(sentence, true_intent, true_pattern, true_args, new_pairs=new_pairs,
                              new_features=new_features)
        self.since_last_train += 1
        return sentence_metrics

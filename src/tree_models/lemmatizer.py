#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk


def tokenize(sentence):
    """Turn a sentence into a set of tokens.
    It used the tokeniser of nltk.

    Args:
        sentence(str): String to tokenize
    
    Returns:
        list: List of tokens
    """
    return nltk.tokenize.word_tokenize(sentence)


def remove_stop_words(sequence, language='english'):
    """Filter stop words from tokens

    Args:
        sequence(list): Tokens from a sentence
    
    Returns:
        list: Filtered sentence
    """
    seq = [word for word in sequence if word not in nltk.corpus.stopwords.words(language)]
    return seq


def map_penn_tree_bank_pos_to_wordnet_pos(treebank_tag):
    """Mapping Penn Treebank POS[1] to WordNet POS tags [2]

    References:
        [1]: List can be found here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        [2]: WordNet POS can be found here: http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

    Args:
        treebank_tag(str): POS Tag of a given word in a sentence
    
    Returns:
        str: Constant of the related WordNet POS
    """
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN
    pass


def lemmatize(sentence, ignore_stop_words=False):
    """Lemmatize a sentence (that is not tokenized).

    This method will use by default nltk word_tokenizer to tokenize a sentence
    We then use into the WordNetLemmatizer.

    Args:
        sentence(str): Raw string to lemmatize
        ignore_stop_words(bool): True if stop words have to be filtered out
    
    Returns:
        list: Sequence of lemmas
    """
    seq = nltk.tokenize.word_tokenize(sentence)
    seq_tokens_pos = nltk.pos_tag(seq)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    if ignore_stop_words:
        seq = remove_stop_words(seq)
        # Filtering stop words in seq token pos
        seq_tokens_pos = [token_pos for token_pos in seq_tokens_pos if token_pos[0] in seq]
    seq = [lemmatizer.lemmatize(w[0], map_penn_tree_bank_pos_to_wordnet_pos(w[1])) for w in seq_tokens_pos]
    return seq

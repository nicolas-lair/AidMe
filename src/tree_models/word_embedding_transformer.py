import torch
import numpy as np
from Tabular_dts import build_vocab

# TODO REWRITE WITH PYTORCH
class WordEmbeddingTransformer:
    """
    Class to compute word_embedding features
    """

    def __init__(self, train, embedding_folder, embedding_list, idf_dict, corpus_size):
        """

        :param train: TabularDataset object, defined in data_handler
        :param embedding_folder: folder with the embedding files
        :param embedding_list: list of the file name of the embedding to be used as features
        :param idf_dict: dictionary of idf of train corpus
        """
        self.idf = idf_dict
        self.corpus_size = corpus_size
        self.embeddings = {}
        for e in embedding_list:
            temp = train
            self.embeddings[e] = build_vocab(temp, embedding_folder, e).vocab

    def sentence_to_embedding(self, data, embed):
        """

        :param data: tokenized data to be processed
        :param embed: embeddinf to use, should be one of self.embedding_list
        :return: torch Tensor of dimension (len(data), EMBED_DIM), each row containing the embedding of a tokenized word
        """
        embedding_list = []
        embedder = self.embeddings[embed]
        for w in data:
            ind = embedder.stoi[w]
            try:
                idf = self.idf[w]
            except KeyError:
                idf = np.log(self.corpus_size)
            vector = embedder.vectors[ind].view(-1, 1) / idf
            embedding_list.append(vector)
        return torch.cat(embedding_list, dim=1)

    def compute_embedding_poolings(self, sentence_matrix):
        """

        :param sentence_matrix: matrix of the embedding of te sentence, given by sentence_to_embedding method
        :return: torch Tensor of dimension 3 * EMBED_DIM
        """
        max_pool = sentence_matrix.max(dim=1)[0]
        min_pool = sentence_matrix.min(dim=1)[0]
        mean_pool = sentence_matrix.mean(dim=1)
        return torch.cat([max_pool, min_pool, mean_pool])

    # TODO check
    def transform(self, data):
        """

        :param data: list of tokenized sentences
        :return: list of torch Tensor, containing the computed features
        """
        result = []
        for d in data:
            aux_list = []
            for e in self.embeddings:
                vectors = self.sentence_to_embedding(d, e)
                aux_list.append(self.compute_embedding_poolings(vectors))
            res = torch.cat(aux_list).numpy()
            result.append(res.reshape(res.shape[0], 1))
        return result

from torch import nn
from torch.nn import functional as func

from .label_encoding import *


class BaseModelNN(nn.Module):
    def __init__(self, sentence_field, emb_dim, output_size, hidden_activation_layer=nn.ReLU(), use_gpu=True):
        super().__init__()
        vocab = sentence_field.vocab

        self.emb_dim = emb_dim
        self.sentence_field = sentence_field
        self.use_gpu = use_gpu

        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.require_grad = False

        output_size_list = list(output_size)
        output_size_list.insert(0, 2 * emb_dim)
        model = nn.ModuleList(
            [nn.Linear(output_size_list[i - 1], output_size_list[i]) for i in range(1, len(output_size_list))])
        for i in range(len(model) - 1, 0, -1):
            model.insert(i, hidden_activation_layer)
        self.model = nn.Sequential(*model)

        self.activation_layer = func.log_softmax

    def pair2embed(self, sent1, sent2):
        def sent2embed(sent):
            x = self.embed(sent)
            x = torch.mean(x, 0)
            return x

        x1 = sent2embed(sent1)
        x2 = sent2embed(sent2)
        sub_ = torch.abs(x1.sub(x2))
        prod_ = torch.mul(x1, x2)
        x = torch.cat((sub_, prod_), 1)
        return x

    def forward(self, sent1, sent2, **kwargs):
        x = self.pair2embed(sent1, sent2)
        x = self.model(x)
        x = self.activation_layer(x, dim=1)
        return x

    def update_embedding_layer(self, vocab):
        self.embed = nn.Embedding(len(vocab), self.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.require_grad = False


class BinModel(BaseModelNN):
    """
    Multilayer Perceptron with two hidden layers, output is the similarity measure called between 0 and 1 (~ probability)
    Activation layer is log_softmax because loss is Kullback Leibler Divergence loss, requiring log probabilites of
    predictors.
    """

    def __init__(self, sentence_field, emb_dim, label_range=(0, 5), output_size=(32, 64, 32, 2),
                 hidden_activation_layer=nn.ReLU(), use_gpu=True):
        super().__init__(sentence_field=sentence_field, output_size=output_size,
                         hidden_activation_layer=hidden_activation_layer, emb_dim=emb_dim, use_gpu=use_gpu)

        self.min_label = label_range[0]
        self.max_label = label_range[1]

    def encode_labels(self, label):
        return encode_binary_labels(label, self.max_label, self.min_label)

    def decode_labels(self, label):
        return decode_binary_labels(label, self.max_label, self.min_label)


class MultiClassModel(BaseModelNN):
    """
    Multilayer Perceptron with two hidden layers, similarity is encoded in n classes and output is n dimension vector
    filled with the probability that the similarity belongs to each class. The similarity measure is then computed with
    the decode_labels method. See the encoding_label method for more explanations.

    Activation layer is log_softmax because loss is Kullback Leibler Divergence loss, requiring log probabilites of
    predictors.
    """

    def __init__(self, sentence_field, emb_dim, label_range=(0, 5), output_size=(64, 32, 6),
                 hidden_activation_layer=nn.ReLU(),
                 use_gpu=True):
        super().__init__(sentence_field=sentence_field, output_size=output_size,
                         hidden_activation_layer=hidden_activation_layer, emb_dim=emb_dim, use_gpu=use_gpu)

        self.min_label = label_range[0]
        self.label_range = label_range[1] - self.min_label
        self.nclass = output_size[-1]

    def encode_labels(self, label):
        lab = (label - self.min_label) / self.label_range * (self.nclass - 1)
        return encode_multiclass_labels(lab, self.nclass)

    def decode_labels(self, label):
        return decode_multiclass_labels(label, self.nclass) / (self.nclass - 1) * self.label_range + self.min_label

    def pair2embed(self, sent1, sent2):
        return super().pair2embed(sent1, sent2)

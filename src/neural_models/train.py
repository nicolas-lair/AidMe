import tqdm
import torch
from scipy.stats import pearsonr
from .eval import compute_predictions
from .utils import transfer_to_gpu


def train_model(model, train_iter, optimizer, loss_func, epochs, val_iter=None, verbose=True, verbose_end=False):
    """
    Train a model
    :param model: pytorch model
    :param train_iter: train iterator, tested with torchtext iterator
    :param val_iter: validation iterator, tested with torchtext Iterator
    :param optimizer: pytorch optimizer ex : torch.optim.Adam
    :param loss_func: pytorch loss function ex : torch.nn.KLDivLoss
    :param epochs: int
    :param verbose: boolean, print running loss during the training
    :param verbose_end: boolean
    :return: tuples of lists containing train and validation lossed and pearson correlation for each epoch
    """
    train_loss = []
    val_loss = []
    train_pearson = []
    val_pearson = []

    data_size = len(train_iter.data())

    if verbose:
        train_iter = tqdm.tqdm_notebook(train_iter)

    for epoch in range(1, epochs + 1):
        log_predictions = []
        labels = []
        running_train_loss = 0
        model.train()  # turn on training mode

        gpu_training = model.use_gpu and torch.cuda.is_available()
        if gpu_training:
            model.cuda()

        for sample in train_iter:
            optimizer.zero_grad()
            model.zero_grad()

            x1, x2, label = sample.sent1, sample.sent2, sample.score
            y = model.encode_labels(label)

            if gpu_training:
                x1, x2, y = transfer_to_gpu((x1, x2, y))

            log_preds = model(x1, x2)
            loss = loss_func(log_preds, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x1.size(0)

            log_predictions.append(log_preds)
            labels.append(label)

        model.eval()
        train_preds, train_labels = torch.exp(torch.cat(log_predictions)).cpu(), torch.cat(labels).data.cpu()

        running_train_pearson = pearsonr(model.decode_labels(train_preds), train_labels)
        running_train_loss /= data_size

        train_loss.append(running_train_loss)
        train_pearson.append(running_train_pearson)

        printed_info = f'''epoch {epoch}, training_loss = {running_train_loss}, training pearson correlation = {running_train_pearson}'''

        if val_iter is not None:
            val_preds, val_labels = compute_predictions(model, val_iter)
            running_val_pearson = pearsonr(val_preds, val_labels)
            running_val_loss = loss_func(torch.log(model.encode_labels(val_preds)),
                                         model.encode_labels(val_labels)
                                         ).item()
            val_loss.append(running_val_loss)
            val_pearson.append(running_val_pearson)
            printed_info += f'''\n validation loss = {running_val_loss}, validation pearson corr = {running_val_pearson}'''

        if verbose:
            print(printed_info)

    if verbose_end and not verbose:
        print(printed_info)

    return train_loss, val_loss, train_pearson, val_pearson


def save_model(model, path):
    """
    Save the parameter of a model
    :param model: Pytorch model
    :param path:
    :return:
    """
    torch.save(model.state_dict(), path)
    print('model saved')


def load_model(initialized_model, path):
    """
    Load the parameters of a pretrained model in a newly initialized model, the pretrained model and new model should be
     exactly similar, same keys and dimensions.
    :param initialized_model: Pytorch model
    :param path:
    :return:
    """
    initialized_model.load_state_dict(torch.load(path))
    print('model_loaded')

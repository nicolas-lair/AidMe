import csv
import io
import os

import torch
from torchtext import vocab
from torchtext.data import Dataset, Example
from torchtext.utils import unicode_csv_reader


class TabularDataset(Dataset):
    def __init__(self, path, format, fields, skip_header=False, quoting=csv.QUOTE_NONE, **kwargs):
        """
        This class overwrite the TabularDataset of torchtext. Adding the quoting parameter in the init function to take
        into account issues with quotes in the data. The file can be wrongly parse and line can be mixed if parameter
        csv.QUOTE_NONE is not used. Below the documentation of original class :

        Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
        """
        make_example = {
            'json': Example.fromJSON,
            'dict': Example.fromdict,
            'tsv': Example.fromCSV,
            'csv': Example.fromCSV
        }[format.lower()]

        self.quoting = quoting  # change from original class

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f,
                                            quoting=self.quoting)  # change from original class
            elif format == 'tsv':
                reader = unicode_csv_reader(
                    f, delimiter='\t',
                    quoting=self.quoting)  # change from original class
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError(
                        'When using a dict to specify fields with a {} file,'
                        'skip_header must be False and'
                        'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(
                    make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


def build_vocab(dataset, embedding_folder, embedding):
    vectors = vocab.Vectors(embedding, embedding_folder)
    data_field = dataset.fields['sent1']
    data_field.build_vocab(dataset, vectors=vectors)
    return data_field


# joblib.dump does the same better
# def dump_sentence_field(sentence_field, folder, name='sentence_field.pkl'):
#     import copy
#     temp_field = copy.copy(sentence_field)
#     temp_field.dtype = None
#     temp_field.tokenize = None
#     path = os.path.join(folder, name)
#     torch.save(temp_field, path)

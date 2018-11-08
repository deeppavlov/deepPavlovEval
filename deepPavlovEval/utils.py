from pathlib import Path
import pandas as pd


class STSReader:
    """
    Data description:
        data_path is directory with two files:
            input.txt
            labels.txt

        input.txt file structure:
            tab-separated pair of sentences on each line
        input.txt file structure:
            floatable number on each line - sentence similarity

    Example:
        input.txt:

        A woman and man are dancing in the rain.	A man and woman are dancing in rain.
        Someone is drawing.	Someone is dancing.

        input.txt:

        5.000
        0.300

    Return:
        {'test': test_set}, where test_set is a list of tuples (sent1, sent2, similarity)
    """
    def read(self, data_path: str, input_fname='input.txt', labels_fname='labels.txt', *args, **kwargs):
        data_path = Path(data_path)

        with open(data_path / input_fname) as f:
            data = [l.rstrip('\n').split('\t') for l in f.readlines()]

        with open(data_path / labels_fname) as f:
            labels = [float(l.rstrip('\n')) for l in f.readlines()]

        merged = [(e[0], e[1], l) for e, l in zip(data, labels)]
        return {'test': merged}


class XNLIReader:
    """
    Data description:
        not_implemented_error
    
    Example:
        not_implemented_error

    Return:
        {'valid': valid_set, 'test': test_set}
        where valid_set and test_set are np.arrays [(sent1, sent2, label), ...]
    """
    def read(self, data_path, valid_fname='xnli.dev.tsv', test_fname='xnli.test.tsv', lang=None):
        data_path = Path(data_path)
        valid = self.read_one(data_path / valid_fname, lang=lang)
        test = self.read_one(data_path / test_fname, lang=lang)
        return {'valid': valid, 'test': test}

    def read_one(self, data_path, lang):
        data = pd.read_csv(data_path, sep='\t')
        data = data[['language', 'gold_label', 'sentence1', 'sentence2']]
        if lang is not None:
            data = data[data.language == lang]

        data = data[['sentence1', 'sentence2', 'gold_label']].values
        return data

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nltk import word_tokenize
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from deeppavlov.dataset_readers.paraphraser_reader import ParaphraserReader
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader

from deepPavlovEval.datautils import STSReader, XNLIReader
from deepPavlovEval.utils import evaluate_embedder_pairwise, evaluate_embedder_clf, evaluate_embedder_nli


class Evaluator:
    def __init__(self, tasks=None, datasets_root=None):
        """
        Class for sentence embeddings evaluations

        Params:
            tasks: subset of ['paraphraser', 'msrvid', 'xnli', 'rusentiment', 'sberfaq']
                   if None (default), use all of them
            datasets_root: path to directory with datasets
                           if None (default), uses ./data directory
        """
        if datasets_root is None:
            self.datasets_root = Path.cwd() / 'data'
        else:
            self.datasets_root = Path(datasets_root)

        if tasks is None:
            tasks = ['paraphraser', 'msrvid', 'xnli', 'rusentiment', 'sberfaq']

        self.task2data = self._load_datasets(self.datasets_root, tasks)
        self.all_results = []

    @staticmethod
    def _load_datasets(datasets_root, tasks):
        task2data = {}
        if 'paraphraser' in tasks:
            task2data['paraphraser'] = ParaphraserReader().read(
                datasets_root / 'Paraphraser')

        if 'msrvid' in tasks:
            task2data['msrvid'] = STSReader().read(
                datasets_root / 'STS2012_MSRvid_translated',
                input_fname='STS.input.MSRvid.txt',
                labels_fname='STS.gs.MSRvid.txt')

        if 'xnli' in tasks:
            task2data['xnli'] = XNLIReader().read(
                datasets_root / 'XNLI-1.0', lang='ru')

        if 'rusentiment' in tasks:
            task2data['rusentiment'] = BasicClassificationDatasetReader().read(
                datasets_root / 'Rusentiment',
                train='rusentiment_preselected_posts.csv',
                test='rusentiment_test.csv',
                x='text',
                y='label')

        # if 'sberfaq' in tasks:
        #     task2data['sberfaq'] = BasicClassificationDatasetReader().read(
        #         datasets_root / 'SBER_FAQ',
        #         train='sber_faq_train.csv',
        #         valid='sber_faq_valid.csv',
        #         test='sber_faq_test.csv',
        #         names=['label', 'text'],
        #         sep='\t',
        #         x='text',
        #         y='label')

        return task2data

    def evaluate(self, embedder, model_name=None):
        """
        Evaluate embedder on tasks

        Params:
            embedder: object with __call__ method which return fixed size
                      batch_size fixed size vectors for batch of sentences
            model_name: used for global results logging
                        default: type(embedder)
        """
        results = {}
        model_name = model_name or type(embedder)

        for task, data in self.task2data.items():
            if task in ['paraphraser', 'msrvid']:
                results[task] = evaluate_embedder_pairwise(embedder, data)
            elif task == 'xnli':
                results[task] = evaluate_embedder_nli(embedder, data)
            else:
                results[task] = evaluate_embedder_clf(embedder, data)

        for task, res in results.items():
            self.all_results.append({'task': task, 'model': model_name, 'metrics': res})

        return results

    def save_results(self, savepath=None):
        if savepath is None:
            savepath = 'results.jsonl'

        with open(savepath, 'w') as outfile:
            for entry in self.all_results:
                json.dump(entry, outfile)
                outfile.write('\n')

    def reset_results(self):
        self.all_results = []

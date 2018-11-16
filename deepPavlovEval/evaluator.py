import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

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
            tasks: subset of ['paraphraser', 'msrvid', 'xnli', 'rusentiment']
                   if None (default), use all of them
            datasets_root: path to directory with datasets
                           if None (default), uses ./data directory
        """
        if datasets_root is None:
            self.datasets_root = Path.cwd() / 'data'
        else:
            self.datasets_root = Path(datasets_root)

        if tasks is None:
            tasks = ['paraphraser', 'msrvid', 'xnli', 'rusentiment']

        self.classification_tasks = ['rusentiment']
        self.paraphraser_tasks = ['paraphraser']
        self.semantic_similarity_tasks = ['msrvid']
        self.pairwise_classification_tasks = ['xnli']

        try:
            self.task2data = self._load_datasets(self.datasets_root, tasks)
        except FileNotFoundError:
            raise FileNotFoundError('Download datasets first')

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

        return task2data

    @staticmethod
    def _ignore_kwarg(kwarg, kwargs):
        if kwarg in kwargs:
            print(f'{kwarg} parameter will be ignored')
            kwargs.pop(kwarg)
        return kwargs
    
    def add_task(self, dataset_dict, task_name, task_type='classification'):
        """
        Add custom task to evaluator.
        
        Params:
            task_name: str, used for results saving and plotting
            dataset_dict: dict, see deepPavlovEval.datautils for more info, depends on task type
            task_type:

                * classification - for one sentence classification
                * pairwise_classification - for two sentences classification (like NLI)
                * paraphraser - for paraphrase detection
                * semantic similarity - for SST-like tasks (similarity scores, not classes)
        """
        self.task2data[task_name] = dataset_dict
        if task_type == 'classification':
            self.classification_tasks.append(task_name)
        elif task_type == 'pairwise_classification':
            self.pairwise_classification_tasks.append(task_name)
        elif task_type == 'paraphraser':
            self.paraphraser_tasks.append(task_name)
        elif task_type == 'semantic_similarity':
            self.semantic_similarity_tasks.append(task_name)
        else:
            raise ValueError(task_type)

    def evaluate(self, embedder, model_name=None, tasks=None):
        """
        Evaluate embedder on tasks

        Params:
            embedder: object with __call__ method which return fixed size
                      batch_size fixed size vectors for batch of sentences
            model_name: used for global results logging
                        default: type(embedder)
            tasks: tasks for evaluation on, default is all
        """
        tasks = tasks or list(self.task2data.keys())
        results = {}
        model_name = model_name or type(embedder)

        for task, data in self.task2data.items():
            if task not in tasks:
                continue
            if task in self.semantic_similarity_tasks:
                results[task] = evaluate_embedder_pairwise(embedder, data)
            elif task in self.paraphraser_tasks:
                results[task] = evaluate_embedder_pairwise(embedder, data, classification=True)
            elif task in self.pairwise_classification_tasks:
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

    def load_results(self, loadpath):
        with open(loadpath, 'r') as infile:
            file_content = infile.read().strip().split('\n')
            results = [json.loads(jline) for jline in file_content]
        self.all_results += results
        return results

    def reset_results(self):
        self.all_results = []

    def plot_results(self, results=None, save=None, show=False, kind='bar', **plot_kwargs):
        """
        Params:
            results: results dict to plot
                     if None (default), plots .all_results
            save: bool or str, save plots to directory with default path (bool) or save (str)
                      default path: results/current_time/
            show: if True, call plt.show()
            plot_kwargs: kwargs for pandas plot function
        """
        if not (save or show):
            raise ValueError('save or show should be specified')

        if save:
            if isinstance(save, str):
                savedir = Path(save)
            else:
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                savedir = Path('results') / current_time
            os.makedirs(savedir, exist_ok=True)
        
        # check plot kwargs
        if 'figsize' in plot_kwargs:
            figsize = plot_kwargs.pop('figsize')
        else:
            figsize = (12, 10)
        plot_kwargs = self.__class__._ignore_kwarg('title', plot_kwargs)

        results = results or self.all_results
        for task in self.task2data.keys():
            all_task_results = [x for x in results if x['task'] == task]
            if not all_task_results:
                print(f'No results for task {task}')
                continue
            to_plot = {res['model']: res['metrics'] for res in all_task_results}
            ax = pd.DataFrame(to_plot).plot(kind=kind, figsize=figsize, title=task, **plot_kwargs)

            if save:
                ax.get_figure().savefig(savedir / f'{task}.png')

        if show:
            plt.show()

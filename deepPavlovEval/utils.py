import numpy as np
from typing import Callable, Sequence, Tuple, Union, List

from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def evaluate_embedder_pairwise(embedder: Callable,
                               dataset_dict: dict,
                               tokenize: bool,
                               classification: bool = False) -> dict:

    similarities, labels = \
        _get_similarities(embedder, dataset_dict['test'], tokenize)

    if not classification:
        results = {'pearson correlation': pearsonr(similarities, labels)[0]}

    else:
        if len(set(labels)) != 2:
            raise RuntimeError('only binary classification is supported')

        if dataset_dict.get('train', []) or dataset_dict.get('valid', []):
            set_name = 'train'
            if not dataset_dict.get('train', []):
                set_name = 'valid'
            similarities_valid, labels_valid = \
                _get_similarities(embedder, dataset_dict[set_name], tokenize)

            _, best_t = _best_score(similarities_valid, labels_valid, f1_score)
            best_f1 = f1_score(labels, similarities > best_t)

            _, best_t = _best_score(similarities_valid, labels_valid, accuracy_score)
            best_acc = accuracy_score(labels, similarities > best_t)
        else:
            best_f1 = f1_score(labels, similarities > .5)
            best_acc = accuracy_score(labels, similarities > .5)
        try:
            roc_auc = roc_auc_score(labels, similarities)
        except Exception as e:
            print('Error while computing roc_auc_score')
            print(e)
            roc_auc = 0

        results = {'f1_best': best_f1,
                   'accuracy_best': best_acc,
                   'roc_auc': roc_auc}

    return results


def evaluate_embedder_clf(embedder: Callable,
                          dataset_dict: dict,
                          tokenize: bool) -> dict:
    embeddings, labels = \
        _get_embeddings(embedder, dataset_dict['train'], tokenize)
    embeddings_test, labels_test = \
        _get_embeddings(embedder, dataset_dict['test'], tokenize)

    results = _get_clf_scores(embeddings, labels, embeddings_test, labels_test)
    return results


def evaluate_embedder_nli(embedder: Callable,
                          dataset_dict: dict,
                          tokenize: bool) -> dict:
    (s1_emb, s2_emb), labels = \
        _get_embeddings_pairwise(embedder, dataset_dict['train'], tokenize)
    features = np.concatenate([s1_emb, s2_emb], 1)

    (s1_emb, s2_emb), labels_test = \
        _get_embeddings_pairwise(embedder, dataset_dict['test'], tokenize)
    features_test = np.concatenate([s1_emb, s2_emb], 1)

    results = _get_clf_scores(features, labels, features_test, labels_test)
    return results


def _best_score(similarities: np.ndarray,
                labels: Sequence,
                target_metric: Callable,
                step: float = 1e-3) -> Tuple[float, int]:
    """
    Search for threshold with target metric
    """
    best_score = 0
    best_t = None
    for t in np.arange(0, 1, step):
        score = target_metric(labels, similarities > t)
        if score >= best_score:
            best_score = score
            best_t = t
    return best_score, best_t


def _maybe_tokenize(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return word_tokenize(x)
    else:
        return x


def _get_embeddings(embedder: Callable,
                    dataset: Sequence,
                    tokenize: bool,
                    batch_size: int = 128,
                    show_progress_bar: bool = True) -> Tuple:
    sents1 = []
    labels = []

    for sent1, label in dataset:
        if tokenize:
            sent1 = _maybe_tokenize(sent1)
        sents1.append(sent1)
        labels.append(label)

    s1_emb = []
    iterator = range(0, len(sents1), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Batches")
    for i in iterator:
        s1_emb.extend(embedder(sents1[i:i+batch_size]))
    return s1_emb, labels


def _get_embeddings_pairwise(embedder: Callable,
                             dataset: Sequence,
                             tokenize: bool,
                             batch_size: int = 128,
                             show_progress_bar: bool = True) -> Tuple:
    sents1 = []
    sents2 = []
    labels = []

    for (sent1, sent2), label in dataset:
        if tokenize:
            sent1 = _maybe_tokenize(sent1)
            sent2 = _maybe_tokenize(sent2)
        sents1.append(sent1)
        sents2.append(sent2)
        labels.append(label)

    s1_emb = []
    s2_emb = []
    iterator = range(0, len(sents1), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Batches")
    for i in iterator:
        s1_emb.extend(embedder(sents1[i:i+batch_size]))
        s2_emb.extend(embedder(sents2[i:i+batch_size]))
    return (s1_emb, s2_emb), labels


def _cosine_similarity(v1: Sequence, v2: Sequence) -> np.ndarray:
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2)

    v1_norm = v1 / np.expand_dims(np.linalg.norm(v1, axis=1), 1)
    v2_norm = v2 / np.expand_dims(np.linalg.norm(v2, axis=1), 1)
    return np.sum(v1_norm * v2_norm, axis=1)


def _get_similarities(embedder: Callable,
                      dataset: Sequence,
                      tokenize: bool,
                      batch_size: int = 128,
                      show_progress_bar: bool = True) -> Tuple:
    similarities, labels = [], []
    iterator = range(0, len(dataset), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Batches")

    for i in iterator:
        (s1_emb, s2_emb), l = _get_embeddings_pairwise(embedder,
                                                       dataset[i:i+batch_size],
                                                       tokenize,
                                                       show_progress_bar=False)
        similarities.extend((s for s in _cosine_similarity(s1_emb, s2_emb)))
        labels.extend(l)

    return similarities, labels


def _get_clf_scores(x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_test: np.ndarray,
                    y_test: np.ndarray) -> dict:

    try:
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        results = {
            'f1(clf_knn)': f1_score(y_test, predictions, average='macro'),
            'accuracy(clf_knn)': accuracy_score(y_test, predictions)
        }
    except Exception as e:
        print('Exception occured while evaluating with KNN')
        print(e)

    model = LinearSVC()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    results.update({
        'f1(clf_svm)': f1_score(y_test, predictions, average='macro'),
        'accuracy(clf_svm)': accuracy_score(y_test, predictions)
    })
    return results

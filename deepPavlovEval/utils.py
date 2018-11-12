import numpy as np

from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def evaluate_embedder_pairwise(embedder, dataset_dict,
                               classification=False, target_metric=accuracy_score):
    similarities, labels = _get_similarities(embedder, dataset_dict['test'])
    results = {'pearson correlation': pearsonr(similarities, labels)[0]}

    if classification:
        if len(set(labels)) != 2:
            raise RuntimeError('only binary classification is supported')

        if dataset_dict.get('train', []) or dataset_dict.get('valid', []):
            set_name = 'train'
            if not dataset_dict.get('train', []):
                set_name = 'valid'
            similarities_valid, labels_valid = _get_similarities(embedder, dataset_dict[set_name])

            _, best_t = _best_score(similarities_valid, labels_valid, f1_score)
            best_f1 = f1_score(labels, similarities > best_t)

            _, best_t = _best_score(similarities_valid, labels_valid, accuracy_score)
            best_acc = accuracy_score(labels, similarities > best_t)
        else:
            best_f1 = f1_score(labels, similarities > .5)
            best_acc = accuracy_score(labels, similarities > .5)
        roc_auc = roc_auc_score(labels, similarities)
        results.update({
            'f1_best': best_f1,
            'accuracy_best': best_acc,
            'roc_auc': roc_auc})

    return results


def evaluate_embedder_clf(embedder, dataset_dict):
    embeddings, labels = _get_embeddings(embedder, dataset_dict['train'])
    embeddings_test, labels_test = _get_embeddings(embedder, dataset_dict['test'])
    
    results = _get_clf_scores(embeddings, labels, embeddings_test, labels_test)
    return results


def evaluate_embedder_nli(embedder, dataset_dict):
    (s1_emb, s2_emb), labels = _get_embeddings_pairwise(embedder, dataset_dict['train'])
    features = np.concatenate([s1_emb, s2_emb], 1)

    (s1_emb, s2_emb), labels_test = _get_embeddings_pairwise(embedder, dataset_dict['test'])
    features_test = np.concatenate([s1_emb, s2_emb], 1)

    results = _get_clf_scores(features, labels, features_test, labels_test)
    return results


def _best_score(similarities, labels, target_metric, step=1e-3):
    """
    Seach for threshold with target metric
    """
    best_score = 0
    best_t = None
    for t in np.arange(0, 1, step):
        score = target_metric(labels, similarities > t)
        if score >= best_score:
            best_score = score
            best_t = t
    return best_score, best_t


def _get_embeddings(embedder, dataset):
    sents1 = []
    labels = []

    for sent1, label in dataset:
        sents1.append(_maybe_tokenize(sent1))
        labels.append(label)

    s1_emb = np.array(embedder(sents1))
    return s1_emb, labels


def _maybe_tokenize(x):
    if isinstance(x, str):
        return word_tokenize(x)
    else:
        return x


def _get_embeddings_pairwise(embedder, dataset):
    sents1 = []
    sents2 = []
    labels = []

    for (sent1, sent2), label in dataset:
        sents1.append(_maybe_tokenize(sent1))
        sents2.append(_maybe_tokenize(sent2))
        labels.append(label)

    s1_emb = np.array(embedder(sents1))
    s2_emb = np.array(embedder(sents2))

    return (s1_emb, s2_emb), labels


def _cosine_similarity(v1, v2):
    v1_norm = v1 / np.expand_dims(np.linalg.norm(v1, axis=1), 1)
    v2_norm = v2 / np.expand_dims(np.linalg.norm(v2, axis=1), 1)
    return np.sum(v1_norm * v2_norm, axis=1)


def _get_similarities(embedder, dataset):
    (s1_emb, s2_emb), labels = _get_embeddings_pairwise(embedder, dataset)
    similarities = _cosine_similarity(s1_emb, s2_emb)

    return similarities, labels

def _get_clf_scores(x_train, y_train, x_test, y_test):
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    results = {
        'knn_f1': f1_score(y_test, predictions, average='macro'),
        'knn_accuracy': accuracy_score(y_test, predictions)
    }
    
    model = LinearSVC()
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    results.update({
        'svm_f1': f1_score(y_test, predictions, average='macro'),
        'svm_accuracy': accuracy_score(y_test, predictions)
    })
    
    return results

from typing import List
from deeppavlov.models.sklearn.sklearn_component import SklearnComponent


class TfidfEmbedder:
    def __init__(self, save_path="vocabs/tfidf.pkl", load_path=None, **kwargs):
        load_path = load_path or save_path

        self.model = SklearnComponent(save_path=save_path,
                                      load_path=load_path,
                                      model_class="sklearn.feature_extraction.text:TfidfVectorizer",
                                      infer_method="transform", **kwargs)

    def __call__(self, batch: List[str]):
        return self.model(batch)

    def save(self, fname: str = None) -> None:
        self.model.save(fname)

    def load(self, fname: str = None):
        self.model.load(fname)
    
    def fit(self, *args):
        self.model.fit(*args)


class CountEmbedder:
    def __init__(self, save_path="vocabs/count.pkl", load_path=None, **kwargs):
        load_path = load_path or save_path

        self.model = SklearnComponent(save_path=save_path,
                                      load_path=load_path,
                                      model_class="sklearn.feature_extraction.text:CountVectorizer",
                                      infer_method="transform",
                                      **kwargs)

    def __call__(self, batch: List[str]):
        return self.model(batch)

    def save(self, fname: str = None) -> None:
        self.model.save(fname)

    def load(self, fname: str = None):
        self.model.load(fname)

    def fit(self, *args):
        self.model.fit(*args)

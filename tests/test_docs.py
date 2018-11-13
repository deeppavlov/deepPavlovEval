from deepPavlovEval import Evaluator
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder

evaluator = Evaluator()
fasttext = FasttextEmbedder('/data/embeddings/wiki.ru.bin', mean=True)
results_fasttext = evaluator.evaluate(fasttext)

import numpy as np
class MyRandomEmbedder:
    def __call__(self, batch):
        return np.random.uniform(size=(len(batch), 300))

my_embedder = MyRandomEmbedder()
results_random = evaluator.evaluate(my_embedder)

all_results = evaluator.all_results

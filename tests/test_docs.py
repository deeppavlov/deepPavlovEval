import unittest

import numpy as np

from deepPavlovEval import Evaluator
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder

class DocsText(unittest.TestCase):

    def test_basic_usage_does_not_fail(self):

        evaluator = Evaluator()

        class MyRandomEmbedder:
            def __call__(self, batch):
                return np.random.uniform(size=(len(batch), 300))

        my_embedder = MyRandomEmbedder()
        results_random = evaluator.evaluate(my_embedder)

        all_results = evaluator.all_results

if __name__ == '__main__':
    unittest.main()

"""
To use BERT encoder, download https://github.com/google-research/bert and show path in PATH_TO_BERT_REPO

This module is not included in __init__ inteitionaly to reduce dependency
"""

import sys
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

# import BERT functions
PATH_TO_BERT_REPO = '../bert'
sys.path.append(PATH_TO_BERT_REPO)
import extract_features as bert


class BERtEmbedder:
    def __init__(self, vocab_file, bert_config_file, init_checkpoint,
                 batch_size=16, max_seq_length=128, mean=True,
                 pooling_strategy='CLS', verbosity=tf.logging.ERROR):
        """
        Parameters:
            vocab_file: bert vocab file
            bert_config_file: bert config file with model parameters
            init_checkpoint: model binary
            batch_size: batch size
            max_seq_length: padding length
            mean: return sentence vectors instead of token vectors
            pooling_strategy: CLS or mean - use vector for [CLS] token or
                              average BERT word vectors (BERT paper for more info)
            verbosity: tf.logging verbose level. Ignored, if None
                       Only errors by default, value=40. Lesser value - more input.

        for batch_size and max_seq_length recommendations see
        https://github.com/google-research/bert#out-of-memory-issues
        """
        assert mean, 'mean == False is not supported yet'
        if verbosity is not None:
            tf.logging.set_verbosity(verbosity)
        
        self.config = bert.modeling.BertConfig.from_json_file(str(bert_config_file))
        self.tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=str(vocab_file), do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy

        self.mean = mean

        self._unique_id = 0

        self.model_fn = bert.model_fn_builder(
          bert_config=self.config,
          init_checkpoint=str(init_checkpoint),
          layer_indexes=(-1,),
          use_tpu=False,
          use_one_hot_embeddings=False)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        run_config = tf.contrib.tpu.RunConfig()
        self.estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=False,
          model_fn=self.model_fn,
          config=run_config,
          predict_batch_size=batch_size)

    def _create_features(self, sentences):
        """Tokenize and add special [CLS] ans [SEP] tokens"""
        examples = []
        for sent in sentences:
            line = bert.tokenization.convert_to_unicode(sent)
            if not line:
                print('WARNING! Blank sentence after unicodifying!')
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                bert.InputExample(unique_id=self._unique_id, text_a=text_a, text_b=text_b))
            self._unique_id += 1

        return bert.convert_examples_to_features(examples, self.max_seq_length, self.tokenizer)

    def _pool(self, layer_output, strategy):
        if strategy.lower() == 'cls':
            return layer_output[0]
        if strategy.lower() == 'mean':
            return np.mean(layer_output, 0)

    def __call__(self, batch):
        """
        Parameters:
            batch: list of sentences (tokenized or not)
                   if sentences are tokenized, detokenization will be preformed
                   because BERT uses sentencepeise tokenizator
        """
        if not isinstance(batch[0], str):
            batch = [' '.join]
        features = self._create_features(batch)

        input_fn = bert.input_fn_builder(
          features=features, seq_length=self.max_seq_length)

        results = []
        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            results.append(result)
        
        embeddings = []
        for result in results:
            layer_output = result['layer_output_0']  # last layer
            if self.mean:
                embedding = self._pool(layer_output, self.pooling_strategy)
            else:
                raise NotImplementedError
            embeddings.append(embedding)

        return embeddings

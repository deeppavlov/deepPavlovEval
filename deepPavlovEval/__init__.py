
__version__ = '0.1'
__author__ = 'Neural Networks and Deep Learning lab, MIPT'
__description__ = 'Sentence embeddings evaluation for russian tasks.'
__keywords__ = ['NLP', 'Embeddings']
__license__ = 'Apache License, Version 2.0'
__email__ = 'info@ipavlov.ai'

# check version
import sys
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'


try:
    from .evaluator import Evaluator
except ImportError:
    'Assuming that requirements are not yet installed'

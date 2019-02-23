from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from bs4 import BeautifulSoup
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class LemmaCountVectorizer(CountVectorizer):
    def __init__(self, stem=False, preprocessing=False, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        # TODO: add all parameters to avoid breaking sklearn, get params method.
        super(LemmaCountVectorizer, self).__init__(input, encoding,
                                                   decode_error, strip_accents,
                                                   lowercase, preprocessor, tokenizer,
                                                   stop_words, token_pattern,
                                                   ngram_range, analyzer,
                                                   max_df, min_df, max_features,
                                                   vocabulary, binary, dtype)
        self.stem = stem
        self.preprocessing = preprocessing

    def build_preprocessor(self):
        preprocessor = super().build_preprocessor()
        if self.preprocessing:
            return lambda doc: self.__remove_noise__(preprocessor(doc))
        return preprocessor

    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        stemmer = PorterStemmer()
        if self.stem:
            return lambda doc: [stemmer.stem(t) for t in tokenize(doc)]
        return tokenize

    def __remove_noise__(self, doc):
        review_text = BeautifulSoup(doc, 'html.parser').get_text()  # remove HTML
        return re.sub("[^a-zA-Z]", " ", review_text)  # remove non-words

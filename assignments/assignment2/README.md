# IMDB Sentiment Analysis
## COMP 551 [project 2](https://cs.mcgill.ca/~wlh/comp551/files/miniproject2_spec.pdf).
IMDb is one of the most popular online databases for movies and personalities, a platform where millions of users read and write movie reviews. This provides a large and diverse dataset for sentiment analysis. In this project, we were tasked to implement different classification models to predict the sentiment of IMDb reviews, either as positive or negative, using only text each review contains. The goal is to find the model with the highest accuracy and generalization. We trained different models using multiple combinations of text features and hyper-parameter settings for both the classifiers and the features, which we found could potentially impact the performance significantly. Every model was evaluated by k-fold cross validation to ensure the consistency of their performance. We found that our best performing model was the Naïve Bayes - Support Vector Machines classifier with bag of words, which reported an accuracy score of **91.880** on the [final test set](https://www.kaggle.com/c/comp-551-imbd-sentiment-classification/leaderboard).


## Project Structure

- _src_ -> contains source code.
    - _data_loader.py_: utility functions to load dataset
    - _nlp_processing.py_: Custom `CountVectorizer` for preprocessing data
    - _naive_bayes.py_: Implementation of Bernoulli Naïve Bayes
    - _nbsvm.py_: Implementation of a variation SVM and NB [[1]](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) and [[2]](https://github.com/Joshua-Chin/nbsvm).
    - _models.ipynb_: Jupyter Notebook with most models implemented.
    - _lstm_imdb.ipynb_: Jupyter Notebook with LSTM NN. 
    **Note:** _We couldn't run this model because of the limitation of GPU, but we aim to work on it to test the accuracy._ 
- _data_ -> this folder is created dinamically and is where datasets and models are stored.
- _test_ -> sanity check of the implementation of Bernoulli Naïve Bayes.
- _data_load.sh_: script to download dataset.
- _make_submission.sh_: script to submit results to Kaggle.

## Installing

### Create environment

`conda create -n newenvironment --file requirements.txt`

### Download data

**Ensure your Kaggle token is in `~/.kaggle`**. You can get a new token in _My Profile->API->Create new API Token_.

`pip install kaggle --upgrade`

Download data set

`sh data_load.sh`

## Running

Once you've downloaded the data, you can reproduce the experiments done using the notebook provided.

At the end, you can make a submission with the best model found, executing `sh make_submisson.sh`


_Note: for executing the stemming, the [Punkt Sentence Tokenizer](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) is necessary. It is downloaded automatically._

## Reproducibility

The model in reported in Kaggle can be found in the section [**Best model in leaderboard**](http://localhost:8888/notebooks/src/models.ipynb#Best-model-in-leaderboard) of the notebook provided.

The model is:

```python
best_pipeline = Pipeline([
    ('vect', LemmaCountVectorizer(analyzer='word', binary=False, decode_error='strict',
            encoding='utf-8', input='content', lowercase=True, max_df=6000, max_features=None, 
            min_df=2, ngram_range=(1, 3), preprocessing=True, preprocessor=None, 
            strip_accents='unicode', token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=nltk.word_tokenize, 
            vocabulary=None, stem=False)),
    ('clf', NBSVM(beta=0.31925992753471094, alpha=1, C=0.40531603281740625, fit_intercept=False))
])
best_pipeline.fit(X_train, y_train)
```
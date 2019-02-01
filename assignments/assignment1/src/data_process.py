from functools import reduce
import os
import pickle
import time
import numpy as np
import csv
from tfidf import buildTFIDFDictionary
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def split_data(data, train_sz=10000, val_sz=1000, test_sz=1000):
    '''Split data into traning, validation and test.'''
    train = data[:train_sz]
    val = data[train_sz:train_sz + val_sz]
    test = data[train_sz + val_sz:]
    return train, val, test


def write_frequent_words(words, file_name):
    '''Save words into a file.'''
    total_frequency = reduce((lambda x, y: x + y), [freq for _, freq in words])
    # Create folder
    dir = os.path.abspath(os.path.join(file_name, os.pardir))
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file_name, 'w') as file_out:
        for word, freq in words:
            # file_out.write("{}:{}\n".format(word, str(freq / total_frequency)))
            file_out.write("{}:{}\n".format(word, freq))


def build_dictionary(data, stop_words=[], dic_size=160, path='../data/words.txt'):
    '''Builds the dictionary using the training set. Then, this dicitionary
       will be used to compute the features for the train, val and test sets'''
    # Count the words in the dataset
    word_counts = {}
    for sample in data:
        # Make the text lower case and split into words
        words = sample['text'].lower().split(' ')
        # TODO: remove punctuation
        words = [word for word in words if word not in stop_words]
        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

    # Select the most frequent words
    frequent_words = [(k, word_counts[k]) for k in sorted(
        word_counts, key=word_counts.get, reverse=True)][:dic_size]

    # Save word frequencies to the file
    write_frequent_words(frequent_words, path)

    # Create dictionary
    dictionary = [w for w, _ in frequent_words]

    # Return the dictionary
    return dictionary


def load_dictionary(file_name='../data/dict.csv'):
    '''Load the feelings dictornary'''
    vec = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            vec.append(row[0])
        print(vec)
    # Return the dictionary
    return vec


def str2vec(str, dic):
    '''String to vector: Given a string and a dictionary returns the vector
    representation of the string.'''
    # Create a vector of zeros of the same size as the dictionary
    vec = np.zeros(len(dic))
    # Split the string into words
    words = str.lower().split(' ')
    # Update the vector with the count of the words
    for word in words:
        if word in dic:
            vec[dic.index(word)] += 1
    return vec


def preprocess_text(data, dictionary, max):
    '''Transform the text features into vectors using the dictionary.'''
    X = np.array([])

    # build feature vector for posts
    for sample in data[:max]:
        feature_vector = str2vec(sample['text'], dictionary)
        if X.size == 0:
            X = np.matrix(feature_vector)
        else:
            X = np.vstack((X, feature_vector))
    return X


def preprocess_boolean(data, col, max):
    # is root feature
    feature = np.zeros((len(data[:max]), 1))
    for i, sample in enumerate(data[:max]):
        if sample[col]:
            feature[i] = 1
    return feature


def preprocess_numeric(raw_data, col, max, f=None):
    # preprocess numbers with some function (i.e. normalize data)
    feature = select_col(raw_data, col, max)
    if f != None:
        feature = f(feature)
    return feature


def select_col(raw_data, col, max):
    y = np.zeros((max, 1))
    for i, post in enumerate(raw_data[:max]):
        y[i] = post[col]
    return y


def func_in_text(raw_data, col, max, f):
    y = np.zeros((max, 1))
    for i, post in enumerate(raw_data[:max]):
        y[i] = f(post[col])
    return y


def preprocess_sentiments(raw_data, col, max, pol):
    y = np.zeros((max, 1))
    sid = SentimentIntensityAnalyzer(lexicon_file='../data/vader_lexicon.txt')

    for i, post in enumerate(raw_data[:max]):
        comment = post[col]
        y[i] = sid.polarity_scores(comment)[pol]
    return y

def min_max(x):
    # TODO: Missing brackets?
    return x - x.min() / x.max() - x.min()


def preprocess(raw_data, feature_list=[], dictionary=[], target='popularity_score', max=0, normalize_func={}):
    """
    Process raw data.
    :param raw_data: data set
    :param dictionary: dictionary of words for text features
    :param feature_list: the resulting dataset will contain only these features
    :param normalize_func: dictionary with normalization function
    :param target:
    :param max: Used to limit the nuimber of samples tu use
    :return:
    """
    # If max is zero use all the samples
    max = len(raw_data) if max == 0 else max

    # Compute the requested features
    features = []
    if 'text' in feature_list:
        features.append(preprocess_text(raw_data, dictionary, max))
    if 'is_root' in feature_list:
        features.append(preprocess_boolean(raw_data, 'is_root', max))
    if 'controversiality' in feature_list:
        f = normalize_func['controversiality'] if 'controversiality' in normalize_func else None
        features.append(preprocess_numeric(raw_data, 'controversiality', max, f))
    if 'children' in feature_list:
        f = normalize_func['children'] if 'children' in normalize_func else None
        features.append(preprocess_numeric(raw_data, 'children', max, f))
    # TODO: Add the two new features here
    if 'square_children' in feature_list:
        features.append(preprocess_numeric(raw_data, 'children', max, lambda x: x ** 2))
    if 'cube_children' in feature_list:
        features.append(preprocess_numeric(raw_data, 'children', max, lambda x: x ** 3))
    if 'fourth_children' in feature_list:
        features.append(preprocess_numeric(raw_data, 'children', max, lambda x: x ** 4))
    if 'log_children' in feature_list:
        features.append(preprocess_numeric(raw_data, 'children', max, lambda x: np.log(x + 0.01)))
    if 'len_text' in feature_list:
        features.append(func_in_text(raw_data, 'text', max, lambda w: len(w)))
    if 'len_sentence' in feature_list:
        features.append(func_in_text(raw_data, 'text', max, lambda words: len(words.split('.'))))
    if 'sentiment_neg' in feature_list:
        features.append(preprocess_sentiments(raw_data, 'text', max, 'neg'))
    if 'sentiment_neu' in feature_list:
        features.append(preprocess_sentiments(raw_data, 'text', max, 'neu'))
    if 'sentiment_pos' in feature_list:
        features.append(preprocess_sentiments(raw_data, 'text', max, 'pos'))
    if 'sentiment_compound' in feature_list:
        features.append(preprocess_sentiments(raw_data, 'text', max, 'compound'))
    # Create the input and target variables
    X = np.matrix([]) if len(features) == 0 else np.hstack(features)
    y = select_col(raw_data, target, max)

    # Return the input and targets
    return X, y


def save_data(train, val, test, id=None):
    '''Save the data in a file with a random name using pickle.'''
    # Define file name
    id = id if id is not None else str(int(time.time() * 100))
    directory = '../data/process/' + id
    # Create folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save files
    pickle.dump(train, open('{}/train-{}.dat'.format(directory, id), 'wb'))
    pickle.dump(val, open('{}/val-{}.dat'.format(directory, id), 'wb'))
    pickle.dump(test, open('{}/test-{}.dat'.format(directory, id), 'wb'))

    # Show filename
    print(" > Data saved in ", directory)


def load_data(id):
    '''Load data from files.'''
    train = pickle.load(open('../data/process/{}/train-{}.dat'.format(id, id), 'rb'))
    val = pickle.load(open('../data/process/{}/val-{}.dat'.format(id, id), 'rb'))
    test = pickle.load(open('../data/process/{}/test-{}.dat'.format(id, id), 'rb'))
    # Return the data
    return train + val + test


def generate_data(data, features, id=None, max=0, stop_words=[], top_words=160, normalize_func={},
                  target='popularity_score', dictType='standard'):
    if os.path.exists("../data/process/" + id):
        print("Configuration ({}) `{}` exists.".format(id, features))
        return
    train, val, test = split_data(data)

    dictionary = []
    if 'text' in features:
        # Build the dictionary
        if (dictType == 'standard'):
            # Build standard dictionary
            print(" > Building the dictionary...")
            dictionary = build_dictionary(train, stop_words=stop_words,
                                          dic_size=top_words,
                                          path="../data/process/%s/words.txt" % id)
        elif (dictType == 'feelings'):
            # Build feelings dictionary
            print(" > Load Feelings dictionary...")
            dictionary = load_dictionary("../data/dict.csv")
        elif (dictType == 'tfidf'):
            # Create sentences
            dictionary = buildTFIDFDictionary(train[0:10000], nWords=160)
            print(" > TFIDF Dictionary: " + str(dictionary))

    train_ = preprocess(train, features, dictionary=dictionary, target=target, max=max, normalize_func=normalize_func)
    test_ = preprocess(test, features, dictionary=dictionary, target=target, max=max, normalize_func=normalize_func)
    val_ = preprocess(val, features, dictionary=dictionary, target=target, max=max, normalize_func=normalize_func)

    print(" > Training: X%s, y%s\tValidation: X%s, y%s\tTesting: X%s, y%s" % (
        train_[0].shape, train_[1].shape, test_[0].shape, test_[1].shape, val_[0].shape, val_[1].shape))

    save_data(train_, val_, test_, id)

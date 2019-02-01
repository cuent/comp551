from functools import reduce
import os

def buildBows(sentences):
    bows = []
    for i, sentence in enumerate(sentences):
        bows.append(sentence.lower().split(' '))
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(data)) +']', end="")
    print()
    return bows

def buildBowsFromData(data):
    import re
    bows = []
    for i, sample in enumerate(data):
        text = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in sample['text'].split("\n")]
        strText = ""
        for line in text:
            strText += line
        bows.append(strText.lower().split(' '))
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(data)) +']', end="")
    print()
    return bows

def buildWordSet(bows):
    wordSet = set()
    for i, bow in enumerate(bows):
        wordSet = wordSet.union(set(bow))
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(bows)) +']', end="")
    print()
    return wordSet

def buildWordDicts(bows, wordSet):
    wordDicts = []
    for i, bow in enumerate(bows):
        wordDict = dict.fromkeys(wordSet, 0)
        for word in bow:
            wordDict[word] += 1
        wordDicts.append(wordDict)
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(bows)) +']', end="")
    print()
    return wordDicts

def computeTF(wordDicts, bows):
    tfDicts = []
    i = 1
    for wordDict, bow in zip(wordDicts, bows):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.items():
            tfDict[word]= count/float(bowCount)
        tfDicts.append(tfDict)
        if ((i)%100==0):
            print ('\r   [' + str(i) + '/' + str(len(bows)) +']', end="")
        i += 1
    print()
    return tfDicts

def computeIDF(sentences):
    import math
    idfDict = {}
    N = len(sentences)

    idfDict = dict.fromkeys(sentences[0].keys(), 0)
    for i, sentence in enumerate(sentences):
        for word, val in sentence.items():
            if val > 0:
                idfDict[word] += 1
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(sentences)) +']', end="")
    print()

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

def computeTFIDF(tfBows, idfs):
    tfidfs = []
    for i, tfBow in enumerate(tfBows):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val*idfs[word]
        tfidfs.append(tfidf)
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(tfBows)) +']', end="")
    print()
    return tfidfs

def combineTFIDFs(tfidfs):
    dictionary = dict.fromkeys(tfidfs[0].keys(), 0)
    for i, tfidf in enumerate(tfidfs):
        for word, val in tfidf.items():
            dictionary[word] += val
        if ((i+1)%100==0):
            print ('\r   [' + str(i+1) + '/' + str(len(tfidfs)) +']', end="")
    print()
    return dictionary

def selectFreqWords(word_counts, nWords, path='../data/words.txt'):
    # Select the most frequent words
    frequent_words = [(k, word_counts[k]) for k in sorted(
        word_counts, key=word_counts.get, reverse=True)][:nWords]

    # Save word frequencies to the file
    write_frequent_words(frequent_words, path)

    # Create dictionary
    dictionary = [w for w, _ in frequent_words]

    # Return the dictionary
    return dictionary

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


def buildTFIDFDictionary(data, nWords=160):
    # Create bows
    print (" > Creating Bag of Words (BOW) for each sample...")
    bows = buildBowsFromData(data)
    # bows = buildBows(data)
    # print (' > Bows: ' + str(bows))

    # Build wordset
    print (" > Creating Word Set from the BOWs...")
    wordSet = buildWordSet(bows)
    # print (' > WordSet: ' + str(wordSet))

    # Build wordDicts
    print (" > Creating Word Dictionaries for each sample...")
    wordDicts = buildWordDicts(bows, wordSet)
    # print (" > WordDicts: " + str(wordDicts))

    # Build TFs
    print (" > Creating Term Frequency (TF) for each sample...")
    tfs = computeTF(wordDicts, bows)
    # print(" > TFs: " + str(tfs))

    # Build IDF
    print (" > Creating Inverse Data Frequency (IDF)...")
    idf = computeIDF(wordDicts)
    # print(" > IDF: " + str(idf))

    # Build TF-IDF
    print (" > Creating Term Frequency - Inverse Data Frequency (TFIDF) for each sample...")
    tfidfs = computeTFIDF(tfs, idf)
    # print(" > TFIDF: " + str(tfidfs))

    # Combine TF-idfs
    print (" > Combining all the TFIDFs...")
    dictionary = combineTFIDFs(tfidfs)
    # print(" > Dictionary: " + str(dictionary))

    # Combine TF-idfs
    print (" > Selecting top TFIDFs...")
    dictionary = selectFreqWords(dictionary, nWords)
    # print(" > Top Dictionary: " + str(dictionary))

    # Ruturn the Dictionary
    return dictionary

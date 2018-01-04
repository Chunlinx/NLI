import numpy as np
from nltk import word_tokenize

def splitToWords(text):
    return word_tokenize(text)

def load_train_data(path):
    f = open(path)
    lines = f.readlines()
    data_count = 0
    max_sequence_length = -1
    label_id = {}
    label_id['neutral'] = 0
    label_id['contradiction'] = 1
    label_id['entailment'] = 2
    label_id['-'] = 3

    word2Id = {}
    word2Id['--end--'] = 0
    word2Id['--unknown--'] = 1
    Id2Word = {}
    Id2Word[0] = '--end--'
    Id2Word[1] = '--unknown--'
    classes = 4

    for i,line in enumerate(lines[1:]):
        # if i == 10000:
        #     break
        if i%10000 == 0:
            print(i)
        splt = line.split('\t')
        sentence_1 = splt[5]
        sentence_2 = splt[6]
        words_1 = splitToWords(sentence_1)
        words_2 = splitToWords(sentence_2)
        if len(words_1) > max_sequence_length or len(words_2) > max_sequence_length:
            max_sequence_length = max([len(words_1),len(words_2)])
        words = words_1 + words_2
        update(word2Id, Id2Word, words)

        data_count += 1

    data_1 = np.zeros([data_count,max_sequence_length],dtype=int)
    data_2 = np.zeros([data_count, max_sequence_length], dtype=int)
    data_length_1 = np.zeros([data_count])
    data_length_2 = np.zeros([data_count])
    labels = np.zeros([data_count,classes])

    for i,line in enumerate(lines[1:]):
        # if i == 10000:
        #     break
        if i%10000 == 0:
            print(i)
        splt = line.split('\t')
        sentence_1 = splt[5]
        sentence_2 = splt[6]
        label = label_id[splt[0]]
        words_1 = splitToWords(sentence_1)
        words_2 = splitToWords(sentence_2)
        data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
        data_2[i, :] = text2Ids(words_2, word2Id, max_sequence_length)
        labels[i,label] = 1
        data_length_1[i] = len(words_1)
        data_length_2[i] = len(words_2)
    result = {}
    result['data_1'] = data_1
    result['data_2'] = data_2
    result['data_length_1'] = data_length_1
    result['data_length_2'] = data_length_2
    result['labels'] = labels
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['total_classes'] = classes
    result['max_sequence_length'] = max_sequence_length
    return result


def load_test_data(path,word2Id,Id2Word,max_sequence_length):
    f = open(path)
    lines = f.readlines()
    data_count = 0
    label_id = {}
    label_id['neutral'] = 0
    label_id['contradiction'] = 1
    label_id['entailment'] = 2
    label_id['-'] = 3

    classes = 4

    for line in lines[1:]:
        splt = line.split('\t')
        sentence_1 = splt[5]
        sentence_2 = splt[6]
        words_1 = splitToWords(sentence_1)
        words_2 = splitToWords(sentence_2)
        if len(words_1) > max_sequence_length or len(words_2) > max_sequence_length:
            max_sequence_length = max([len(words_1),len(words_2)])
        words = words_1 + words_2
        update(word2Id, Id2Word, words)
        data_count += 1

    data_1 = np.zeros([data_count,max_sequence_length],dtype=int)
    data_2 = np.zeros([data_count, max_sequence_length], dtype=int)
    data_length_1 = np.zeros([data_count])
    data_length_2 = np.zeros([data_count])
    labels = np.zeros([data_count,classes])




    for i,line in enumerate(lines[1:]):
        splt = line.split('\t')
        sentence_1 = splt[5]
        sentence_2 = splt[6]
        label = label_id[splt[0]]
        words_1 = splitToWords(sentence_1)
        words_2 = splitToWords(sentence_2)
        data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
        data_2[i, :] = text2Ids(words_2, word2Id, max_sequence_length)
        labels[i,label] = 1
        data_length_1[i] = len(words_1)
        data_length_2[i] = len(words_2)

    result = {}
    result['data_1'] = data_1
    result['data_2'] = data_2
    result['data_length_1'] = data_length_1
    result['data_length_2'] = data_length_2
    result['labels'] = labels
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['total_classes'] = classes

    return result



def update(word2Id,Id2Word,words):
    keys = list(word2Id.keys())
    for word in words:
        if word not in keys:
            word2Id[word] = len(keys)
            Id2Word[len(keys)] = word
            keys.append(word)

def text2Ids(words,word2Id,max_sequence_len):
    a = np.zeros(max_sequence_len,dtype=int)
    keys = word2Id.keys()
    for i,word in enumerate(words):
        if word in keys:
            a[i] = word2Id[word]
        else:
            a[i] = word2Id['--unknown--']
    return a
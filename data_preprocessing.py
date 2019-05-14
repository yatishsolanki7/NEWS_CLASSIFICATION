import pandas as pd
import numpy as np
import glob
import pickle
import timeit
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
ps = nltk.PorterStemmer()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import random

def read_data(path):

    print('fetching data')

    start = timeit.default_timer()

    # list containing names of all groups or newsGroups = classes
    news_groups = [ x[len(path):] for x in glob.glob(path+'*')]

    # dictionary:  key = NewsGroup value=list of all news of the group
    news_groups_dict={}

    group_count=1
    for g in news_groups:
        print(group_count,' ',g)
        news_groups_dict[g]=[]
        c=1
        for f in glob.glob(path + g + '/*'):
            dff = pd.read_table(f, encoding='windows-1252', header=None, sep='delimiter', engine='python')
            # print(group_count,' ',g,f[len(path)+len(g)+1:],c,)
            news_groups_dict[g].append(dff)
            c+=1
        group_count+=1

    with open('Data/news_groups_dict.pickle', 'wb') as f:
        pickle.dump(news_groups_dict, f)

    stop = timeit.default_timer()
    print('news_groups_dict.pickle created successfully. execution time = ' , (stop - start) / 60 , ' mins')

    return news_groups_dict

# takes dataframe of news and preprocess it
def clean_news(news):
    processed_news = news[0].str.lower()
    processed_news = processed_news.str.replace(r'\S+@\S+', ' ')  # remove email
    processed_news = processed_news.str.replace(r'[0-9]+', ' ')  # remove numbers
    processed_news = processed_news.str.replace(r'[^\w\d\s]', ' ')  # remove punctuation
    processed_news = processed_news.str.replace(r'\s+', ' ')  # remove whitespace
    processed_news = processed_news.str.strip()  # remove trailing leading spaces
    processed_news = processed_news.apply(
        lambda x: [term for term in word_tokenize(x) if term not in stop_words])  # remove stop words
    processed_news = processed_news.apply(
        lambda x: [ps.stem(term) for term in x])  # Remove word stems using a Porter stemmer
    news[0] = processed_news


def clean_all_news(news_groups_dict):
    start = timeit.default_timer()
    group_count=1
    for g in news_groups_dict.keys():
        print(group_count,' cleaning ',g)
        for news in news_groups_dict[g]:
            clean_news(news)
        group_count+=1

    with open('Data/news_groups_dict.pickle','wb') as f:
        pickle.dump(news_groups_dict,f)

    stop = timeit.default_timer()
    print('cleaned news_groups_dict.pickle created successfully. execution time = ',(stop-start)/60,' mins')


def find_lexicons(news_groups_dict):
    all_words = []
    group_count=1
    for g in news_groups_dict.keys():
        print(group_count,' processing ',g)
        for news in news_groups_dict[g]:
            for line in news[0]:
                for word in line:
                    all_words.append(word)
        group_count+=1
    uniq_words = nltk.FreqDist(all_words)
    lexicons = [x for x in uniq_words.keys() if 700 <= uniq_words[x] <= 7000]
    return lexicons

def vectorize_news(news,lexicons):
    vector=np.zeros(len(lexicons))
    for line in news[0]:
        for word in line:
            if word in lexicons:
                index_value = lexicons.index(word)
                vector[index_value] += 1
    return vector

def vectorize_all(news_groups_dict,lexicons):
    features = []
    group_count=1
    for g in news_groups_dict.keys():
        print(group_count,' vectorizing ',g)
        for news in news_groups_dict[g]:
            features.append((vectorize_news(news,lexicons),group_count))
        group_count+=1
    return features

def one_hot_encode_GroupName(features):
    label_encoded = le.fit_transform(features[:, 1])
    onehot_encoded = ohe.fit_transform(label_encoded.reshape(-1, 1))
    for i in range(len(features)):
        features[i, 1] = onehot_encoded[i]


def get_features(news_groups_dict):
    lexicons = find_lexicons(news_groups_dict)
    features = vectorize_all(news_groups_dict, lexicons)
    features = np.array(features)
    # one_hot_encode_GroupName(features)
    return features


def split_test_data(features, test_size=0.2):
    testing_size = int(test_size * len(features))

    X_train = list(features[:, 0][:-testing_size])
    y_train = list(features[:, 1][:-testing_size])
    X_test = list(features[:, 0][-testing_size:])
    y_test = list(features[:, 1][-testing_size:])

    return X_train, y_train, X_test, y_test


def main_fun(path):

     news_groups_dict = read_data(path)
     clean_all_news(news_groups_dict)
     # with open('NLP/NewsClassification/Data/news_groups_dict.pickle', 'rb') as f:
     #     news_groups_dict = pickle.load(f)

     features = get_features(news_groups_dict)

     random.shuffle(features)

     X_train, y_train, X_test, y_test = split_test_data(features,0.2)

     with open('Data/processed_training_data_temp.pickle', 'wb') as f:
         pickle.dump([X_train, y_train, X_test, y_test], f)
         print("processed_training_data.pickle created successfully")


main_fun('Data/20_newsgroups/')




import re
import nltk
import pandas as pd
from textblob import Word
import numpy as np
import os
import csv
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, log_loss
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import reuters

nltk.download('returs')
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups


def extract_bbc_data():
    data_folder = 'resources/bbc'
    folders = ['business', 'entertainment', 'politics', 'sport', 'tech']
    os.chdir(data_folder)
    x = []
    y = []
    for i in folders:
        files = os.listdir(i)
        for text_file in files:
            file_path = i + "/" + text_file
            print("reading file:", file_path)
            with open(file_path) as f:
                data = f.readlines()
            data = ''.join(data)
            x.append(data)
            y.append(i)
    data = {'category': y, 'text': x}
    df = pd.DataFrame(data)
    print('writing csv file...')
    df.to_csv('bbc_raw.csv', index=False)
    df = pd.read_csv('bbc_raw.csv')
    data = df.sample(frac=1)
    print(data);
    print(len(data))





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    extract_bbc_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

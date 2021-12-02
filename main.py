import re
import nltk
import pandas as pd
from textblob import Word
import numpy as np
import os
import csv
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score, log_loss
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#from google.colab import files, drive

from sklearn import naive_bayes
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import reuters
#nltk.download('reuters')
from sklearn.datasets import fetch_20newsgroups

################# Load Data ####################
################################################

def load_reuters():
    documents = reuters.fileids()
    train_docs_id = list(filter(lambda doc:doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    train_labels = ([reuters.categories(doc_id)[0] for doc_id in train_docs_id])
    test_lables = ([reuters.categories(doc_id)[0] for doc_id in test_docs_id])
    
    data = train_docs + test_docs
    cat = train_labels + test_lables
    
    df = pd.DataFrame({'text':data, 'category':cat})
    a = df.groupby('category').size()


    for i in range(len(df)):
        cat = df.loc[i,'category']
        if a[cat]<50:
            df = df.drop(index=i)
    df.reset_index(drop=True, inplace= True)
    data = df
    return data

def load_news():
    newsgroups = fetch_20newsgroups()
    df = {'text':newsgroups.data,'category':newsgroups.target}
    data = pd.DataFrame(df)
    #print(data)
    return data

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
            with open(file_path) as f:
                data = f.readlines()
            data = ''.join(data)
            x.append(data)
            y.append(i)
    data = {'category': y, 'text': x}
    df = pd.DataFrame(data)
    #print('writing csv file...')
    df.to_csv('bbc_raw.csv', index=False)
    df = pd.read_csv('bbc_raw.csv')
    data = df.sample(frac=1)
    return data

###########################################
###########################################

def main():
    print("Loading Ruters Data....\n")
    reuters_df = load_reuters()
    print("Done\nLoading 20 newsgroups Data...")
    newsgroup_data = load_news()
    print("Done\nLoading BBC data...")
    bbc_data = extract_bbc_data()
    print("Done\n")



if __name__ == '__main__':
    main()
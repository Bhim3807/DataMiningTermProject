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

import pickle
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


#############~~Preprocessing~~#############
###########################################

def clean_str(string):
    string = re.sub(r"\’s", "", string)
    string = re.sub(r"\’ve", "", string)
    string = re.sub(r"n\’t", "", string)
    string = re.sub(r"\’re", "", string)
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\’ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"’", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\’\‘]", " ",string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def prep(data):
    x = data['text'].tolist()
    for index,value in enumerate(x):
        x[index] = " ".join([Word(word).lemmatize() for word in clean_str(value).split() if word not in set(stopwords.words('english'))])
    return pd.DataFrame({'text':x,'category':data['category']})

##########################################
###########################################


#############~~Training~~##################
###########################################

def classify(data):
    x = data['text'].tolist()
    y = data['category'].tolist()

    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(x)
    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer.fit(word_count_vector)
    vect = TfidfVectorizer(stop_words='english', min_df=2)
    X = vect.fit_transform(x)
    Y = np.array(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = True)
    modelDT = tree.DecisionTreeClassifier()
    modelDT.fit(X_train, Y_train)
    model_pred_DT = modelDT.predict(X_test)

    modelKNN = KNeighborsClassifier(n_neighbors=20)
    modelKNN.fit(X_train, Y_train)
    model_pred_KNN = modelKNN.predict(X_test)

    modelRF = RandomForestClassifier(n_estimators=1000, random_state=0)
    modelRF.fit(X_train, Y_train)
    model_pred_RF = modelRF.predict(X_test)

    modelNB = naive_bayes.MultinomialNB()
    modelNB.fit(X_train, Y_train)
    model_pred_NB = modelNB.predict(X_test)

    modelsvm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    modelsvm.fit(X_train, Y_train)
    model_pred_svm = modelsvm.predict(X_test)

    modelLR = LogisticRegression(verbose=1, solver='liblinear',
    random_state=0, C=5, penalty='l2',max_iter=1000)
    modelLR.fit(X_train, Y_train)
    model_pred_LR = modelLR.predict(X_test)
    print("\nDecision Tree Classifier accuracy: " , accuracy_score(Y_test, model_pred_DT) * 100, "%.")
    print("Decision Tree Classifier macro f1 avg: " , f1_score(Y_test,model_pred_DT, average='macro') * 100, "%.")
    print("K-Nearest Neighbor Classifier accuracy: " , accuracy_score(Y_test, model_pred_KNN) * 100, "%.")
    print("K-Nearest Neighbor Classifier macro f1 avg: " , f1_score(Y_test, model_pred_KNN, average='macro') * 100, "%.")
    print("Random Forest Classifier accuracy: " , accuracy_score(Y_test, model_pred_RF) * 100, "%.")
    print("Random Forest Classifier macro f1 avg: " , f1_score(Y_test,model_pred_RF, average='macro') * 100, "%.")
    print("Naive Bayes Classifier accuracy: " , accuracy_score(Y_test,model_pred_NB) * 100, "%.")
    print("Naive Bayes Classifier macro f1 avg: " , f1_score(Y_test,model_pred_NB, average='macro') * 100, "%.")
    print("SVM Classifier accuracy: " , accuracy_score(Y_test,model_pred_svm) * 100, "%.")
    print("SVM Classifier macro f1 avg: " , f1_score(Y_test,model_pred_svm, average='macro') * 100, "%.")
    print("Logistic Regression accuracy: " , accuracy_score(Y_test,model_pred_LR) * 100, "%.")
    print("Logistic Regression macro f1 avg: " , f1_score(Y_test,model_pred_LR, average='macro') * 100, "%.")

    print("\n-----Decision Tree Classifier:-----\nAccuracy = " ,
    accuracy_score(Y_test, model_pred_DT) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_DT))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_DT))

    print("\n-----K-Nearest Neighbor Classifier:-----\nAccuracy = " ,
    accuracy_score(Y_test, model_pred_KNN) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_KNN))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_KNN))

    print("\n-----Random Forest Classifier:-----\nAccuracy = " ,
    accuracy_score(Y_test, model_pred_RF) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_RF))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_RF))

    print("\n-----Naive Bayes Classifier:-----\nAccuracy: " ,
    accuracy_score(Y_test, model_pred_NB) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_NB))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_NB))

    print("\n-----SVM Classifier-----\nAccuracy: " , accuracy_score(
    Y_test, model_pred_svm) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_svm))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_svm))

    print("\n-----Logistic Regression-----\nAccuracy: " ,
    accuracy_score(Y_test, model_pred_LR) * 100, "%.")
    print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_pred_LR))
    print("\nClassification Report: \n", classification_report(Y_test,model_pred_LR))

    result_acc = [(accuracy_score(Y_test, model_pred_DT), 'DT', modelDT
    ), (accuracy_score(Y_test, model_pred_KNN), 'KNN', modelKNN), (
    accuracy_score(Y_test, model_pred_RF), 'RF', modelRF), (
    accuracy_score(Y_test, model_pred_NB), 'NB', modelNB), (
    accuracy_score(Y_test, model_pred_svm), 'SVM', modelsvm), (
    accuracy_score(Y_test, model_pred_LR), 'LR', modelLR)]
    return result_acc


def main():
    '''
    print("Loading Ruters Data....\n")
    reuters_df = load_reuters()
    print("Done\nLoading 20 newsgroups Data...")
    newsgroup_data = load_news()
    print("Done\nLoading BBC data...")
    bbc_data = extract_bbc_data()
    print("Done\n")
    
    print("Preprcessing Data : String Cleaning, lemmatization, stopwords")
    print("Ruters......")
    reuters_df = prep(reuters_df)
    pd.to_pickle(reuters_df, "./reuters_df.pkl")

    print("20 newsgroups......")
    newsgroup_data = prep(newsgroup_data)
    pd.to_pickle(newsgroup_data, "./newsgroup_data.pkl")
    
    print("BBC......")
    bbc_data = prep(bbc_data)
    pd.to_pickle(bbc_data, "./bbc_data.pkl")
    '''

    reuters_df = pd.read_pickle("resources/bbc/reuters_df.pkl")
    newsgroup_data = pd.read_pickle("resources/bbc/newsgroup_data.pkl")
    bbc_data = pd.read_pickle("resources/bbc/bbc_data.pkl")
    classify(bbc_data)


if __name__ == '__main__':
    main()
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
import json

import pickle
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn import naive_bayes
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import imblearn

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
            print("reading file:", file_path)
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




#############~~Sampling~~##################
###########################################

def sampling(data,typ):
    x = data['text']
    y = data['category']
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(x)
    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer.fit(word_count_vector)
    vect = TfidfVectorizer(stop_words='english', min_df=2)
    X = vect.fit_transform(x)
    Y = np.array(y)

    if typ=="oversampling":
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_train_new, y_train_new = oversample.fit_resample(X, Y)
        return X_train_new,y_train_new

    elif typ =="under_sampling":
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_train_new, y_train_new = undersample.fit_resample(X, Y)
        return X_train_new,y_train_new
    else:
        oversample = SMOTE()
        X_train_new, y_train_new = oversample.fit_resample(X, Y)
        return X_train_new,y_train_new


#############~~Training~~##################
###########################################

def classify(x,y, sampling):
    X = x
    Y = y
    if(sampling==False):
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

    f1 = {'DT':f1_score(Y_test,model_pred_DT, average='macro') * 100, 'KNN':f1_score(Y_test, model_pred_KNN, average='macro') * 100,'RF':f1_score(Y_test,model_pred_RF, average='macro') * 100,'NB':f1_score(Y_test,model_pred_NB, average='macro') * 100,'SVM':f1_score(Y_test,model_pred_svm, average='macro') * 100,'LR':f1_score(Y_test,model_pred_LR, average='macro') * 100}

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
    return f1


def main():
    '''
    print("Loading Ruters Data....\n")
    reuters_df = load_reuters()
    print("Done\nLoading 20 newsgroups Data...")
    newsgroup_data = load_news()
    print("Done\nLoading BBC data...")
    bbc_data = extract_bbc_data("bbc")
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
    bbc_imb = pd.read_pickle("resources/bbc/bbc_imb.pkl")

    raw = {'BBC':{},'Ruters':{},'20_Newsgroup':{}, 'BBC_org':{}}

    raw['BBC'] = classify(bbc_imb['text'],bbc_imb['category'], False)
    raw['Ruters'] = classify(reuters_df['text'], reuters_df['category'], False)
    raw['20_Newsgroup'] = classify(newsgroup_data['text'], newsgroup_data['category'], False)
    raw['BBC_org'] = classify(bbc_data['text'], bbc_data['category'], False)
   

    with open('raw.json', 'w') as f:
        json.dump(raw, f)

    oversample = {'BBC':{},'Ruters':{},'20_Newsgroup':{}, 'BBC_org':{}}
    reuters_oversample = sampling(reuters_df,"oversampling")
    oversample['Ruters'] = classify(reuters_oversample[0],reuters_oversample[1],True)

    bbc_oversample = sampling(bbc_imb,"oversampling")
    oversample['BBC'] = classify(bbc_oversample[0],bbc_oversample[1],True)

    newsgroup_oversample = sampling(newsgroup_data,"oversampling")
    oversample['20_Newsgroup'] = classify(newsgroup_oversample[0],newsgroup_oversample[1],True)

    bbc_org_oversample = sampling(bbc_data,"oversampling")
    oversample['BBC_org'] = classify(bbc_org_oversample[0],bbc_org_oversample[1],True)

    with open('oversample.json', 'w') as f:
        json.dump(oversample, f)

    undersample = {'BBC':{},'Ruters':{},'20_Newsgroup':{}, 'BBC_org':{}}
    reuters_undersample = sampling(reuters_df,"under_sampling")
    undersample['Ruters'] = classify(reuters_undersample[0],reuters_undersample[1],True)

    bbc_undersample = sampling(bbc_imb,"under_sampling")
    undersample['BBC'] = classify(bbc_undersample[0],bbc_undersample[1],True)

    newsgroup_undersample = sampling(newsgroup_data,"under_sampling")
    undersample['20_Newsgroup'] = classify(newsgroup_undersample[0],newsgroup_undersample[1],True)

    bbc_org_undersample = sampling(bbc_data,"under_sampling")
    undersample['BBC_org'] = classify(bbc_org_undersample[0],bbc_org_undersample[1],True)

    with open('undersample.json', 'w') as f:
        json.dump(undersample, f)




if __name__ == '__main__':
    main()
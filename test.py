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

modelsvm = svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’)
modelsvm.fit(X_train, Y_train)
model_pred_svm = modelsvm.predict(X_test)

modelLR = LogisticRegression(verbose=1, solver=’liblinear’,
random_state=0, C=5, penalty=’l2’,max_iter=1000)
modelLR.fit(X_train, Y_train)
model_pred_LR = modelLR.predict(X_test)
print("\nDecision Tree Classifier accuracy: " , accuracy_score(Y_test, model_pred_DT) * 100, "%.")
print("Decision Tree Classifier macro f1 avg: " , f1_score(Y_test,model_pred_DT, average=’macro’) * 100, "%.")
print("K-Nearest Neighbor Classifier accuracy: " , accuracy_score(Y_test, model_pred_KNN) * 100, "%.")
print("K-Nearest Neighbor Classifier macro f1 avg: " , f1_score(Y_test, model_pred_KNN, average=’macro’) * 100, "%.")
print("Random Forest Classifier accuracy: " , accuracy_score(Y_test, model_pred_RF) * 100, "%.")
print("Random Forest Classifier macro f1 avg: " , f1_score(Y_test,model_pred_RF, average=’macro’) * 100, "%.")
print("Naive Bayes Classifier accuracy: " , accuracy_score(Y_test,model_pred_NB) * 100, "%.")
print("Naive Bayes Classifier macro f1 avg: " , f1_score(Y_test,model_pred_NB, average=’macro’) * 100, "%.")
print("SVM Classifier accuracy: " , accuracy_score(Y_test,model_pred_svm) * 100, "%.")
print("SVM Classifier macro f1 avg: " , f1_score(Y_test,
model_pred_svm, average=’macro’) * 100, "%.")
print("Logistic Regression accuracy: " , accuracy_score(Y_test,model_pred_LR) * 100, "%.")
print("Logistic Regression macro f1 avg: " , f1_score(Y_test,model_pred_LR, average=’macro’) * 100, "%.")

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

result_acc = [(accuracy_score(Y_test, model_pred_DT), ’DT’, modelDT
), (accuracy_score(Y_test, model_pred_KNN), ’KNN’, modelKNN), (
accuracy_score(Y_test, model_pred_RF), ’RF’, modelRF), (
accuracy_score(Y_test, model_pred_NB), ’NB’, modelNB), (
accuracy_score(Y_test, model_pred_svm), ’SVM’, modelsvm), (
accuracy_score(Y_test, model_pred_LR), ’LR’, modelLR)]
return result_acc
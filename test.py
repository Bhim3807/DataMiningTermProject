import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

data ={}
with open('undersample.json') as f:
    data = json.load(f)

labels = ['BBC', 'Ruters', '20_newsgroup', 'BBC_original']

DT = []
KNN = []
RF = []
NB= []
SVM = []
LR = []

for i in data.keys():
	DT.append(data[i]['DT']/100)
	KNN.append(data[i]['KNN']/100)
	RF.append(data[i]['RF']/100) 
	NB.append(data[i]['NB']/100)
	SVM.append(data[i]['SVM']/100) 
	LR.append(data[i]['LR']/100) 
print(DT)

df = pd.DataFrame({'Decision Tree': DT,
                    'KNN': KNN,'Random Forest':RF,'Naive Bayes':NB,'SVM':SVM,'Logistic Regression':LR}, index=labels)
ax = df.plot.bar(rot=0, color={"Decision Tree": "green", "KNN": "red","Random Forest":"blue","Naive Bayes":"yellow","SVM":"purple","Logistic Regression":'black'})
ax.legend(loc='best', fontsize=7)
plt.ylabel("F1 Scores")
plt.title("Undersampling")
plt.tight_layout()
plt.show()
'''
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 4*width/2, DT, width, label='Decision Tree')
rects2 = ax.bar(x - 3*width/2, KNN, width, label='KNN')
rects3 = ax.bar(x - 2*width/2, RF, width, label='Random Forest')
rects4 = ax.bar(x + 2*width/2, NB, width, label='Naive Bayes')
rects5 = ax.bar(x + 3*width/2, SVM, width, label='SVM')
rects6 = ax.bar(x + 4*width/2, LR, width, label='Logistic Regression')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 scores')
ax.set_title('Original data')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)
ax.bar_label(rects5, padding=3)
ax.bar_label(rects6, padding=3)

#fig.tight_layout()

plt.show()
'''
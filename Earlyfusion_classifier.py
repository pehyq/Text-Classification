# encoding=utf8

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


data_dir = './data' 

print("Loading Labels...")
with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding = "utf-8") as f:
	y = np.array(f.readlines())

print("Loading Text...")
with open(os.path.join(data_dir, 'EF_samples_processed.txt'), 'r', encoding = "utf-8") as f:
	x = f.readlines()
    

print("Extract features...")
x_feats = TfidfVectorizer().fit_transform(x)
print(x_feats.shape)


print("Start training and predict...")
kf = KFold(n_splits=10)
# Naive Bayes Classifier
avg_p = 0
avg_r = 0
avg_f1 = 0
for train, test in kf.split(x_feats):
    model = MultinomialNB().fit(x_feats[train], y[train]) 
    predicts = model.predict(x_feats[test])
    avg_p	+= precision_score(y[test],predicts, average='macro')
    avg_r	+= recall_score(y[test],predicts, average='macro')
    avg_f1 += f1_score(y[test],predicts, average='macro')

print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))
print('Average f1 score is %f.' %(avg_f1/10.0))

# KNN Classifier
avg_p = 0
avg_r = 0
avg_f1 = 0
for train, test in kf.split(x_feats):
    model = KNeighborsClassifier().fit(x_feats[train], y[train]) 
    predicts = model.predict(x_feats[test])
    avg_p	+= precision_score(y[test],predicts, average='macro')
    avg_r	+= recall_score(y[test],predicts, average='macro')
    avg_f1 += f1_score(y[test],predicts, average='macro')

print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))
print('Average f1 score is %f.' %(avg_f1/10.0))

# Randomforest Classifier
avg_p = 0
avg_r = 0
avg_f1 = 0
for train, test in kf.split(x_feats):
    model = RandomForestClassifier().fit(x_feats[train], y[train]) 
    predicts = model.predict(x_feats[test])
    avg_p	+= precision_score(y[test],predicts, average='macro')
    avg_r	+= recall_score(y[test],predicts, average='macro')
    avg_f1 += f1_score(y[test],predicts, average='macro')

print('Average Precision is %f.' %(avg_p/10.0))
print('Average Recall is %f.' %(avg_r/10.0))
print('Average f1 score is %f.' %(avg_f1/10.0))


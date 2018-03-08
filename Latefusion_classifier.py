# encoding=utf8

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

#################### Load pre-processed data ########################

data_dir = './data' 

print("Loading Labels...")
with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding = "utf-8") as f:
	y = np.array(f.readlines())

print("Loading Text...")
with open(os.path.join(data_dir, 'samples_processed.txt'), 'r', encoding = "utf-8") as f_text:
	x_text = f_text.readlines()

print("Loading user descriptions...")
with open(os.path.join(data_dir, 'description_processed.txt'), 'r', encoding = "utf-8") as f_description:
	x_description = f_description.readlines()

print("Loading hashtags...")
with open(os.path.join(data_dir, 'hashtags_processed.txt'), 'r', encoding = "utf-8") as f_hashtag:
	x_hashtag = f_hashtag.readlines()
    
print("Loading location features...")
with open(os.path.join(data_dir, 'geo_processed.txt'), 'r', encoding = "utf-8") as f_geo:
	x_geo = f_geo.readlines()

print("Extract features...")
x_text_feats = TfidfVectorizer().fit_transform(x_text)
x_description_feats = TfidfVectorizer().fit_transform(x_description)
x_hashtag_feats = TfidfVectorizer().fit_transform(x_hashtag)
x_geo_feats = TfidfVectorizer().fit_transform(x_geo)
print(x_text_feats.shape)
print(x_description_feats.shape)
print(x_hashtag_feats.shape)
print(x_geo_feats.shape)

print("Start training and predict...")
kf = KFold(n_splits=10)

#################### Optimising different ML model ########################

#Model_test = [MultinomialNB(), KNeighborsClassifier(), RandomForestClassifier()]
#
## Classifier for text
#for modelx in Model_test:
#    text_pred = []
#    avg_p = 0
#    avg_r = 0
#    for train, test in kf.split(x_text_feats):
#        model = modelx.fit(x_text_feats[train], y[train]) 
#        predicts = model.predict(x_text_feats[test])
#        #print(classification_report(y[test],predicts))
#        avg_p	+= precision_score(y[test],predicts, average='macro')
#        avg_r	+= recall_score(y[test],predicts, average='macro')
#
#    print('Average Precision text is %f.' %(avg_p/10.0))
#    print('Average Recall text is %f.' %(avg_r/10.0))
#
#
## Classifier for user description
#for modelx in Model_test:
#    description_pred = []
#    avg_p = 0
#    avg_r = 0
#    for train, test in kf.split(x_description_feats):
#    	model = modelx.fit(x_description_feats[train], y[train]) 
#    	predicts = model.predict(x_description_feats[test])
#    	#print(classification_report(y[test],predicts))
#    	avg_p	+= precision_score(y[test],predicts, average='macro')
#    	avg_r	+= recall_score(y[test],predicts, average='macro')
#    
#    print('Average Precision description is %f.' %(avg_p/10.0))
#    print('Average Recall description is %f.' %(avg_r/10.0))
#
#
## Classifier for hash tags
#for modelx in Model_test:   
#    hashtag_pred = []
#    avg_p = 0
#    avg_r = 0
#    for train, test in kf.split(x_hashtag_feats):
#    	model = modelx.fit(x_hashtag_feats[train], y[train]) 
#    	predicts = model.predict(x_hashtag_feats[test])
#    	#print(classification_report(y[test],predicts))
#    	avg_p	+= precision_score(y[test],predicts, average='macro')
#    	avg_r	+= recall_score(y[test],predicts, average='macro')
#    
#    print('Average Precision hashtag is %f.' %(avg_p/10.0))
#    print('Average Recall hashtag is %f.' %(avg_r/10.0))
#
#
## Classifier for location
#for modelx in Model_test:  
#    geo_pred = []
#    avg_p = 0
#    avg_r = 0
#    for train, test in kf.split(x_geo_feats):
#    	model = modelx.fit(x_geo_feats[train], y[train]) 
#    	predicts = model.predict(x_geo_feats[test])
#    	#print(classification_report(y[test],predicts))
#    	avg_p	+= precision_score(y[test],predicts, average='macro')
#    	avg_r	+= recall_score(y[test],predicts, average='macro')
#    
#    print('Average Precision location is %f.' %(avg_p/10.0))
#    print('Average Recall location is %f.' %(avg_r/10.0))


# Based on the optimisation, the best model for Text, description, hashtags is Niave Bayes and location is random forest

##################### Train final model ###############################
# Classifier for text
text_pred = np.empty([0, 10])
for train, test in kf.split(x_text_feats):
    model = MultinomialNB().fit(x_text_feats[train], y[train]) 
    predicts = model.predict_proba(x_text_feats[test])
    text_pred = np.concatenate((text_pred, predicts))


# Classifier for user description
description_pred = np.empty([0, 10])
for train, test in kf.split(x_description_feats):
    model = MultinomialNB().fit(x_description_feats[train], y[train]) 
    predicts = model.predict_proba(x_description_feats[test])
    description_pred = np.concatenate((description_pred, predicts))

# Classifier for hash tags
hashtag_pred = np.empty([0, 10])
for train, test in kf.split(x_hashtag_feats):
    model = MultinomialNB().fit(x_hashtag_feats[train], y[train]) 
    predicts = model.predict_proba(x_hashtag_feats[test])
    hashtag_pred = np.concatenate((hashtag_pred, predicts))


# Classifier for location 
geo_pred = np.empty([0, 10])
for train, test in kf.split(x_geo_feats):
    modelG = RandomForestClassifier().fit(x_geo_feats[train], y[train]) 
    predicts = modelG.predict_proba(x_geo_feats[test])
    geo_pred = np.concatenate((geo_pred, predicts))


# Assign different weights to different classifer
grid = np.array([(T, D, H, G) for T in np.arange(0,1,0.05) for D in np.arange(0,1,0.05) for H in np.arange(0,1,0.05) for G in np.arange(0,1,0.05)])

grid_new = []
for i in range(len(grid)):
    if sum(grid[i]) == 1:
        grid_new.append(grid[i])

precisions = []
recalls = []
f1_scores = []
y = np.float64(y)

for i in range(len(grid_new)):
    combine_pred = grid_new[i][0]*text_pred + grid_new[i][1]*description_pred + grid_new[i][2]*hashtag_pred + grid_new[i][3]*geo_pred
    prediction = np.apply_along_axis(np.argmax, 1, combine_pred)
    
    avg_p = precision_score(y, prediction, average='macro')
    avg_r = recall_score(y, prediction, average='macro')
    avg_f1 = f1_score(y, prediction, average='macro')

    precisions.append(avg_p)
    recalls.append(avg_r)
    f1_scores.append(avg_f1)

opt_id = np.argmax(f1_scores)
print(grid_new[opt_id])
print('Optimal Precision is %f.' %precisions[opt_id])
print('Optimal Recall is %f.' %recalls[opt_id])
print('Optimal F1 score is %f.' %f1_scores[opt_id])

##################### Train model with only 1 addtional feature ###############################
# only description
#grid = np.array([(T, D) for T in np.arange(0,1,0.1) for D in np.arange(0,1,0.1)])
# only hashtag
#grid = np.array([(T, H) for T in np.arange(0,1,0.1) for H in np.arange(0,1,0.1)])
# only location
#grid = np.array([(T,G) for T in np.arange(0,1,0.1) for G in np.arange(0,1,0.1)])
#
#grid_new = []
#for i in range(len(grid)):
#    if sum(grid[i]) == 1:
#        grid_new.append(grid[i])
#
#precisions = []
#recalls = []
#f1_scores = []
#y = np.float64(y)
#
#for i in range(len(grid_new)):
#    combine_pred = grid_new[i][0]*text_pred + grid_new[i][1]*description_pred 
#    combine_pred = grid_new[i][0]*text_pred + grid_new[i][1]*hashtag_pred 
#    combine_pred = grid_new[i][0]*text_pred + grid_new[i][1]*geo_pred 
#    prediction = np.apply_along_axis(np.argmax, 1, combine_pred)
#    
#    avg_p = precision_score(y, prediction, average='macro')
#    avg_r = recall_score(y, prediction, average='macro')
#    avg_f1 = f1_score(y, prediction, average='macro')
#
#    precisions.append(avg_p)
#    recalls.append(avg_r)
#    f1_scores.append(avg_f1)
#
#opt_id = np.argmax(f1_scores)
#print(grid_new[opt_id])
#print('Optimal Precision is %f.' %precisions[opt_id])
#print('Optimal Recall is %f.' %recalls[opt_id])
#print('Optimal F1 score is %f.' %f1_scores[opt_id])
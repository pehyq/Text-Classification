# Text-Classification
Text Classification of Twitter microblog data

### Objectives:
To develop a classifier to classify the incoming microblog streams into 10 pre-defined classes (“Health”, “Business”, “Arts”, “Sports”, “Shopping”, “Politics”, “Education”, “Technology”, “Entertainment”, “Travel”). The classification needs to examine the content of microblogs, as well as other information such as hashtags and user info.

Data set is not uploaded due to privacy issues, to contact me if interested.

### Program structure
The programme consists of the following:
1.	Pre-processing of text data
2.	Feature selection based on Document frequency
3.	Classification using machine learning techniques
4.	Testing and optimisation based on 10-fold cross validation and precision, recall, and f1 scores

### Environment Setting
Python 3.6.3 |Anaconda custom (64-bit)| (default, Oct 15 2017, 03:27:45) [MSC v.1900 64 bit (AMD64)]

### Installation 
1. nltk 
2. simplejson
3. numpy
4. scipy
5. scikit-learn

### Usage 
### Late fusion
1. Run 'Latefusion_preprocess.py' to prepocess tweet content, including data cleaning (e.g., remove url, punctuations, time), word tokenize, stemming, remove low-frequency words and stopping words. 
 
2. Run 'Latefusion_classifier.py' which performs the following:
- load preprocessed data
- extract features
- optimise different ML techniques for each features (commented)
- train final model
- outputs optimal precision, recall and f1 score of the best model

### Early Fusion
1. Run 'Earlyfusion_preprocess.py' to prepocess tweet content, including data cleaning (e.g., remove url, punctuations, time), word tokenize, stemming, remove low-frequency words and stopping words.
 
2. Run 'Earlyfusion_classifier.py' which performs the following:
- load preprocessed data
- extract features
- optimise different ML techniques for each features (commented)
- train final model
- outputs precision, recall and f1 score of the NB, KNN and RF model in order

#!/usr/bin/env python
# coding: utf-8

# # 1. load the data set

# In[1]:


import sklearn
import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv("/home/abz/Desktop/sms-spam-collection-dataset/spam.csv",encoding = "latin-1")
df= df[['v1', 'v2']]
df= df.rename(columns = {'v1': 'label', 'v2': 'text'})


# In[3]:


df.info()
df.head()


# # 2. Cleaning data

# In[4]:


#1. Removing Punctuationtrain

import string
string.punctuation

def remove_punctuation(txt):
    txt_nopunct="".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


# In[5]:


df['newtext']=df['text'].apply(lambda x: remove_punctuation(x))
df.head()


# In[6]:


#2. tokenization of data
import re

def tokenize(txt):
    tokens=re.split('\W+', txt)
    return tokens
df['token_text']=df['newtext'].apply(lambda x: tokenize(x.lower()))


# In[7]:


df.head()


# In[8]:


#3. remove stop words

import nltk
stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenize):
    clean=[word for word in tokenize if word not in stopwords]
    return clean

df['stop_clean']=df['token_text'].apply(lambda x: remove_stopwords(x))


# In[9]:


df.head()


# In[10]:


#4. stemming

from nltk.stem import PorterStemmer
ps=PorterStemmer()

def stemming(txt):
    words=[ps.stem(word) for word in txt]
    return words

df['stem_words']=df['stop_clean'].apply(lambda x: stemming(x))
df.head()


# In[11]:


#5. lemmatization

def lammatization(txt):
    lam=[wn.lammetize(word) for word in txt]
    return lam

df['lam_words']=df['stem_words'].apply(lambda x: stemming(x))
df.head()


# In[12]:


#as our "lam_words column is column of lists, and not text. 
#Tfidf Vectorizer works on text so convert this column into string"

df['lam_words']=[" ".join(review) for review in df['lam_words'].values]


# # 3. split data into sets

# In[62]:


from sklearn import model_selection, naive_bayes, svm
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['lam_words'],df['label'],test_size=0.2)


# # 4. encoding 

# In[63]:


from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# # 5. Word Vectorization

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['lam_words'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[72]:


#print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)


# # 6. Use the ML Algorithms to Predict the outcome

# 1.Naive Bayes Classifier 

# In[73]:


from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[74]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)


# In[75]:


# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# 2. Support Vector Machine

# In[76]:


# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[ ]:





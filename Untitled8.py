#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd
train_data = pd.read_csv(r"C:\Users\Sai Sathwik\Downloads\train.csv", encoding='ISO-8859-1')
test_data = pd.read_csv(r"C:\Users\Sai Sathwik\Downloads\test.csv", encoding='ISO-8859-1')
print(train_data.head())
print(test_data.head())
train_missing_values = train_data.isnull().sum()
test_missing_values = test_data.isnull().sum()
train_missing_values, test_missing_values
sentiment_distribution = train_data['sentiment'].value_counts()
plt.figure(figsize=(8, 5))
sentiment_distribution.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title("Sentiment Distribution in Train Dataset", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
from sklearn.impute import SimpleImputer
columns_to_impute = [col for col in train_data.columns if col != "selected_text"]
train_data_imputed = train_data.copy()
test_data_imputed = test_data.copy()
imputer = SimpleImputer(strategy='most_frequent')
train_data_imputed[columns_to_impute] = imputer.fit_transform(train_data_imputed[columns_to_impute])
test_data_imputed = pd.DataFrame(imputer.transform(test_data_imputed), columns=test_data.columns)
import re
from sklearn.feature_extraction.text import TfidfVectorizer
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply the cleaning function to the text columns
train_data_imputed['text'] = train_data_imputed['text'].apply(clean_text)
test_data_imputed['text'] = test_data_imputed['text'].apply(clean_text)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data_imputed['text'])
X_test = vectorizer.transform(test_data_imputed['text'])

# Prepare the target variable
y_train = train_data_imputed['sentiment']
y_test = test_data_imputed['sentiment']
# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predictions and evaluation
y_pred_logreg = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")
print(classification_report(y_test, y_pred_logreg))


# In[14]:


from sklearn.svm import SVC

# Train an SVM model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy}")
print(classification_report(y_test, y_pred_svm))
from sklearn.naive_bayes import MultinomialNB

# Train a Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {nb_accuracy}")
print(classification_report(y_test, y_pred_nb))


# In[ ]:





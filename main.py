"""
Importing necessary packages
"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


import datetime
# for large and multidimensional arrays
import re
import numpy as np

# for data manipulation and analysis
import pandas as pd

# Natural Language processing tool-kit
import nltk

# stopwords corpus
from nltk.corpus import stopwords

# Stemmer
from nltk.stem import PorterStemmer

# for Bag of Words Vector
from sklearn.feature_extraction.text import CountVectorizer

# For TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# for word2Vec
from gensim.models import word2vec


"""
Loading Datasets
"""
df = pd.read_csv("amazon-fine-food-reviews/Reviews.csv")


"""
Exploratory Data Analysis
"""
# See first 10 datas from the dataset
print(df.head(10))

# Find all the columns
print(df.columns)

# See  values having positive score
print(df[df["Score"] > 3])


"""
Data Preprocessing
"""
# Step 1: Sorting date for tie based splitting for model  train and test dataset
df["Time"] = df["Time"].map(lambda t: datetime.datetime.fromtimestamp(
    int(t)).strftime('%Y-%m-%d %H:%M:%S'))

df_sorted = df.sort_values(
    'ProductId',  kind="quicksort", ascending=True)
# Step 2: Data Cleansing : Removing duplicate datas

# Reviews might contain duplicate entries. So, we need to remove the duplicate entries so that weget unbiased data for analysis.
final_df = df_sorted.drop_duplicates(
    subset=("UserId", "ProfileName", "Time", "Text"))

# Step 3 : Helpfullness numerator should always be less than helpfullness denominator
"""
Helpfullness numerator is the amount of user who found the review helpful.
Helpfulness denomimantor is the amount of users whether they found the review useful or not.
"""
final = final_df[final_df["HelpfulnessNumerator"]
                 <= final_df["HelpfulnessDenominator"]]

final_X = final["Text"]
final_Y = final["Score"]

# Step 4 : Converting all the words in the text to lowercase and removing punctuations or html tags if any.
# Also we perform:
"""
Stemming : It is the process of converting a word into its base/root word. This will help to reduce vectore dimension as we don't consider all the similar words.For ex: cats to cat, playing to play etc.
Stopword : It is the process of removing those unnecessary words from the text which when removed doesn't change the sentiment of the text . For ex: Ball is green => Ball green
"""
tmp = []
snow_stemmer = nltk.stem.SnowballStemmer(language="english")
for sentence in final_X:
    sentence = sentence.lower()  # converting the sentence to lowercase
    clean = re.compile("<.*?>")
    sentence = re.sub(clean, " ", sentence)  # remove html tags
    sentence = re.sub(r"[?|!|\'|\"|#]", r"", sentence)
    sentence = re.sub(r"[.|,|:|(|)|\|/]", r" ",
                      sentence)  # removing puntuations

    words = [
        snow_stemmer.stem(word)
        for word in sentence.split()
        if word not in stopwords.words("english")
    ]  # removing stopwords and then stemming the result
    tmp.append(words)

final_X = tmp

# print(final_X[1])

# preparing sentences from list of words
sent = []
for row in final_X:
    sequence = " "
    for word in row:
        sequence = sequence + word
    sent.append(sequence)

final_X = sent
# print(final_X[1])

"""
Encoding Techniques : BOW
"""
count_vect = CountVectorizer(max_features=5000)
binary_bow_data = count_vect.fit_transform(binary=True)

"""
Scaling data
"""
final_bow_np = StandardScaler(with_mean=False).fit_transform(binary_bow_data)

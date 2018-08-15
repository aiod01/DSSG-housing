from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import Common.nlp_utils as nutils
import pandas as pd
import numpy as np
import re

sum =0
def bag_words (row, colname):
    if len(row[colname]) > 0 and row[colname][0] != "nan" :
        try:
           return vec.fit_transform(row[colname])
        except:
            global sum
            sum+=1

# define input file path here:
f = "../results/Standardized_Deduped_Datasets/1000samples_20180810-JL_partial_labels_standardized_bedroomscaled.csv"

df = pd.read_csv(f)
vec = CountVectorizer(stop_words=None)

# Missing values
#df['sqft'] = df['sqft'].dropna().apply(lambda x: float(re.sub("\D", "", x)))

df['lat'].fillna(-1, inplace=True)
df['long'].fillna(-1, inplace=True)
df['price'].fillna(-1, inplace=True)
df['sqft'].fillna(-1, inplace=True)
df['rooms'].fillna(-1, inplace=True)
df['description'].fillna("", inplace=True)
df['title'].fillna("", inplace=True)

rooms = pd.get_dummies(df['rooms'])
df = pd.concat([df, rooms], axis=1)
print(df.head())
print(df.info())
# # Bag of words
# desc_corpus = vec.fit_transform(train['description'])
#
# title_corpus = vec.fit_transform(train['title'])
# title_corpus = title_corpus.toarray()

# features = ['price', 'sqft', 'rooms']
features = ['price', 'sqft', -1.0, 0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.1]

train = df.iloc[:99]
test = df.iloc[99:]

clf = RandomForestClassifier(n_jobs=2, random_state=1234)
clf.fit(train[features], train['Label_0.entire_1.part'])
prediction = clf.predict(test[features])
test['predicted'] = prediction

#test.to_csv("../results/Standardized_Deduped_Datasets/1000samples_20180810_pred.csv")

scores = cross_val_score(clf, train[features], train['Label_0.entire_1.part'], cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf.fit(train[features], train['Label_0.entire_1.suite_2.room'])
predict2 = clf.predict(test[features])
test['predict2'] = predict2
scores2 = cross_val_score(clf, train[features], train['Label_0.entire_1.suite_2.room'], cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

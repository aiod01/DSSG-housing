from sklearn.feature_extraction import text
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pandas as pd


# define input file path here:
# f = "../results/Standardized_Deduped_Datasets/1000samples_20180815_withoutstar_labelledJA.csv"
f = "../results/Standardized_Deduped_Datasets/1000samples_20180815_labelledJA.csv"

df = pd.read_csv(f)
df['lat'].fillna(-1, inplace=True)
df['long'].fillna(-1, inplace=True)
df['price'].fillna(-1, inplace=True)
df['sqft'].fillna(-1, inplace=True)
df['rooms'].fillna(-1, inplace=True)
df['description'].fillna("", inplace=True)
df['title'].fillna("", inplace=True)
df['Category'].fillna("", inplace=True)
df['Category_text'] = df['Category'].apply(lambda x: str(x))

vec = text.TfidfVectorizer(max_df=0.7)

train, test = train_test_split(df, test_size=0.2)
train_x = vec.fit_transform(train['title'])
test_x = vec.transform(test['title'])

model = LogisticRegression()
model.fit(train_x, train["Category_text"])
test['predict'] = model.predict(test_x)
scores = metrics.accuracy_score(test['Category_text'],test['predict'])
c_val_score = cross_val_score(model, train_x, train['Category_text'], cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (c_val_score.mean(), c_val_score.std() * 2))
print(scores)

#features = ['price', 'sqft', 0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
model2 = RandomForestClassifier()
clf = RandomForestClassifier(n_jobs=2, random_state=1234)
clf.fit(train_x, train['Category_text'])
predict2 = clf.predict(test_x)
test['predict2'] = predict2
scores2 = cross_val_score(clf, train_x, train['Category_text'], cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

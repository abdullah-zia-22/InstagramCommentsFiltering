import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

train_data = pd.read_csv("dataset (davidson).csv")

c=train_data['class']
train_data.rename(columns={'tweet' : 'text',
                   'class' : 'category'}, 
                    inplace=True)
a=train_data['text']
b=train_data['category'].map({0: 'hate_speech', 1: 'offensive_language',2: 'neither'})

train_data= pd.concat([a,b,c], axis=1)
train_data.rename(columns={'class' : 'label'}, 
                    inplace=True)
train_data

train_data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train_data['text']]

train_data

train_data['category_id'] = train_data['category'].factorize()[0]

category_id_df = train_data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

category_to_id

train_data.head()



tfidfVector = TfidfVectorizer(sublinear_tf=True, min_df=50, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidfVector.fit_transform(train_data['text_lem']).toarray()
labels = train_data.category_id
features.shape

X_train = features
Y_train = labels


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()

cv_df.groupby('model_name').accuracy.mean()


model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train_data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

sentence = ["looking beautiful"]
text_features = tfidfVector.transform(sentence)

predictions = model.predict(text_features)
for text, predicted in zip(sentence, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")
#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))


# to be imported
from keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import nltk
from __future__ import print_function
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras.layers import SpatialDropout1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding

# Read the input dataset
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

train_data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train_data['text']]

train_data['category_id'] = train_data['category'].factorize()[0]

from io import StringIO
category_id_df = train_data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)
train_data.tail(10)

y = train_data['label'].values
x = train_data['text_lem'].values

x.shape

# encode the text with word sequences - Preprocessing step 1
tk = Tokenizer(num_words= 200, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")
tk.fit_on_texts(x)
x = tk.texts_to_sequences(x)
x = sequence.pad_sequences(x, maxlen=200)

print(x.shape)

# Label Encoding categorical data for the classification category
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
print(y[:150])
y.shape
#print(np.unique(y, return_counts=True))

# Perform one hot encoding 

y = to_categorical(y, num_classes= 3)

print(y.shape)

# Data cleaning

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, train_data.index, test_size=0.33, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = sequence.pad_sequences(x_train, maxlen=200)
x_test = sequence.pad_sequences(x_test, maxlen=200)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

max_features = 24783
maxlen = 200
embedding_dims=50


 
print('Build model...')
model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test,y_test))

new_complaint = ['you are not beautiful. you are a black women trying to be white']
seq = tk.texts_to_sequences(new_complaint)
#labels = ['hate speech:0','offensive:1','neither:2']
padded = sequence.pad_sequences(seq, maxlen=200)
pred =np.argmax(model.predict(padded), axis=-1)

print(pred)

model.save('my_model.h5')

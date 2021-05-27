# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:33:26 2021

@author: Sanjay
"""

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras import models
from keras.layers import Dense


# Load Data set
df = pd.read_csv('Reviews.csv')
df.head()

# Required dataset for training
""" We need only two columns Content and Ratings.Overall """
reqData = df[['Content', 'Ratings.Overall']]
reqData.head()

# Data Exploration
reqData.info()
reqData.describe()

# Check For Null values
reqData.isnull().sum()
""" There is no null values """

# Check for unique values
reqData['Ratings.Overall'].unique()

# Data Visualization
reqData['Ratings.Overall'].value_counts()
num = [5,4,3,2,1]
plt.bar(num, reqData['Ratings.Overall'].value_counts())
""" We can see that most of the people give 5 Rating """

reqData.Content[2]

# Data Preprocessing / Cleaning
sw = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
ps.stem('jumping')

# Create a function to clean data
def cleanText(sample):
    sample = sample.lower()
    # Remove punctuations
    sample = re.sub("[^a-zA-Z]+", ' ', sample)
    # Tokenizing sample (split every word)
    sample = sample.split()
    # Remove stopwords
    sample = [s for s in sample if s not in sw] # List comprehension
    # Stemmimg (convert words to root words: jumping-jump) 
    sample = [ps.stem(s) for s in sample]
    # Lemmatization it reduces size of vocabolary.
    sample = [lemmatizer.lemmatize(s) for s in sample]
    # Joins words
    sample = " ".join(sample)
       
    return sample
    
cleanText(reqData.Content[2])

# Apply cleanText function to DataFrame
reqData['CleanedContent'] = reqData['Content'].apply(cleanText)
reqData.head()

corpus = reqData['CleanedContent'].values #  Convert to Array
y = reqData['Ratings.Overall']


# Convert Text data to matrix formate or tabular formate
cv = CountVectorizer(ngram_range=(1, 3), max_df=0.5, max_features=30000)
tfidf = TfidfTransformer()

X = cv.fit_transform(corpus)
print(X[0])
X = tfidf.fit_transform(X)
print(X[0])

X.shape
y.shape

# Create  Artificial Neural Network
model = models.Sequential([
    Dense(128, input_shape=(2020,30000,), activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="selu"),
    Dense(6, activation="softmax"),
    ])

model.summary()

#  Compile Model
model.compile(
    optimizer="rmsprop",
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy'],
    )

# Split data into train, test and validation
xval = X[:500]
xtest = X[500:900]
xtrain = X[900:]

yval = y[:500]
ytest = y[500:900]
ytrain = y[900:]

# Converted data into Array
xtrainArr = xtrain.toarray()
xtestArr = xtest.toarray()
xvalArr = xval.toarray()

ytrainArr = np.array(y[900:])
yvalArr = np.array(y[:500])
ytestArr = np.array(y[500:900])


hist = model.fit(xtrainArr, ytrainArr, batch_size=16, epochs=2, validation_data=(xvalArr, yvalArr))

result = hist.history

# Training Visualization
plt.plot(result['val_accuracy'], label='Val_acc')
plt.plot(result['accuracy'], label="train_acc")
plt.legend()
plt.show()


plt.plot(result['val_loss'], label='Val_loss')
plt.plot(result['loss'], label="train_los")
plt.legend()
plt.show()
""" Here we can adjust our epochs values to overcome model overfitting"""

model.evaluate(xvalArr, yvalArr)

# Test Our Model
yPred = model.predict(xtestArr)

ypreValues =[]
for i in range(0, 400): 
    val = np.argmax(yPred[i])
    ypreValues.append(val)
    
yPredArr = np.array(ypreValues).reshape(-1, 1)
ytestArr = ytestArr.reshape(-1, 1)
ids = np.arange(400).reshape(-1, 1)
finalMatrix = np.hstack((ids, ytestArr, yPredArr))

# Create a final result csv file
final =pd.DataFrame(finalMatrix, columns=['Id','Original_Value','Predicted_value'])  
# final.to_csv('Rating_Comperision.csv', index=False)
print('completed')

# utilities :
import nltk
nltk.download("stopwords")
nltk.download('punkt')
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#----------------------------------------------------------------------------------------------------------------#
# data

# Import dataset from CSV file
df = pd.read_csv('twitter_training.csv')
print(df.describe())
print(df.dtypes)
print(df.info())

print(df.columns)
print('shape of data is : ', df.shape)
print(df['sentiment'].unique())
# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows with null or NaN values
df.dropna(inplace=True)

# Print the cleaned dataset
print(df.head())
print(df.tail())

# Check for NaN values
nan_values = df.isnull().sum()
print("NaN values:\n", nan_values)

# statistiques and informations
print(df.describe())
print(df.dtypes)
print(df.info())

print(df.columns)
print('shape of data is : ', df.shape)
print(df['sentiment'].unique())


#tweet content cleaning and data processing
text_df = df.drop(['entity','sentiment'], axis=1)
print(text_df.head())

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

text_df['text'] = text_df['Tweet_content,'].apply(data_processing)


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

text_df['Tweet_content,'] = text_df['Tweet_content,'].apply(lambda x: stemming(x))

#applications :
def polarity(text):
    return TextBlob(text).sentiment.polarity

text_df['polarity'] = text_df['text'].apply(polarity)


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

text_df['sentiment'] = text_df['polarity'].apply(sentiment)
print(text_df.tail(20))

# visualtion of our data :
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = df)
plt.savefig("mygraph.png")

#visualtion of data after implying sentiment function above

fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)
plt.savefig("mygraph2.png")

#-------------------------------------------------------------------------------------------------------------------------------#

# modele :
vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])
 
X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

import warnings
warnings.filterwarnings('ignore')

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)

print("Test accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))

style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()
plt.savefig('mygraph3.png')

 #improvement grad numbre 1
from sklearn.model_selection import GridSearchCV
print("gridsearch")

param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)

print("Best parameters:", grid.best_params_)

y_pred = grid.predict(x_test)

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy numbre 2 : {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

 #improvement grad numbre 2

from sklearn.svm import LinearSVC
print("LinearSVC")

SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)

svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy numbre 3 : {:.2f}%".format(svc_acc*100))

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred)) 

#improvement grad number 3
print("3333333333")
grid = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':["linear","poly","rbf","sigmoid"],
    'degree':[1,3,5,7],
    'gamma':[0.01,1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)

print("Best parameter:", grid.best_params_)

y_pred = grid.predict(x_test)

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy 4: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))





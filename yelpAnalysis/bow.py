import pandas as pd
import joblib
df = pd.read_csv("data/train_yelp_60k.csv")  # Load your dataset
test_dataframe = pd.read_csv("data/test_yelp_60k.csv")  # Load your dataset
df.head()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

train_df = df.copy()
test_df = test_dataframe.copy()

train_df.info()
print(train_df)

# Encode the class variable/column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Class'] = le.fit_transform(train_df['Class'])
print(train_df['Class'])

print(dict(zip(le.classes_, range(len(le.classes_)))))

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

#Take out stop words and do stemmer thingy to neutralize word
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

numRows = len(train_df)

for i in range(numRows):
  review = train_df.loc[i, 'Text']
  review = review.lower()
  review = word_tokenize(review)
  stopWords.discard('not')
  review = [word for word in review if word not in stopWords]
  review = [stemmer.stem(word) for word in review]
  train_df.loc[i, 'Text'] = ' '.join(review)

# max_features - Limit the vocab to that number of unique words
# min_df ( minimum document frequency) - Removes the number of words that appear less than that number in documents.
# max_df ( maximum document frequency) - Removes the number of words that appear more than tha number in documents.

vectorizer = CountVectorizer(max_features=55000,min_df=100, max_df=0.9);
X = vectorizer.fit_transform(train_df['Text'])
X_train_df = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())


train_df = pd.concat([train_df, X_train_df], axis=1)
train_df = train_df.drop(columns=['ID', 'Text'])

# Print the final DataFrame
print(train_df)

# PRE-PROCESSING TEST

numRowsTest = len(test_df)

for i in range(numRowsTest):
  reviewTest = test_df.loc[i, 'Text']
  # print(review)
  reviewTest = reviewTest.lower()
  reviewTest = word_tokenize(reviewTest)
  stopWords.discard('not')
  reviewTest = [word for word in reviewTest if word not in stopWords]
  reviewTest = [stemmer.stem(word) for word in reviewTest]
  test_df.loc[i, 'Text'] = ' '.join(reviewTest)

XTest = vectorizer.transform(test_df['Text'])
X_test_df = pd.DataFrame.sparse.from_spmatrix(XTest, columns=vectorizer.get_feature_names_out())


test_df = pd.concat([test_df, X_test_df], axis=1)
testIDs = test_df['ID']
test_df = test_df.drop(columns=['ID','Text'])

# Print the final DataFrame
print(test_df)

#Feature Selection - Train
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = train_df.drop(columns=['Class'])
y = train_df['Class']

selector = SelectKBest(chi2, k=5000)  # Select top 5000 features
X_selected = selector.fit_transform(X, y)



X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
bestAcc = 0
bestModelName = None
bestModel = None



logisticRegression = LogisticRegression(random_state=42, max_iter=1000)
NeuralNetwork = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu',max_iter=1000,early_stopping=True,random_state=42)

models = {
    "Neural Network": NeuralNetwork,
    "Logistic Regression": logisticRegression
    # "Decision Trees": decisionTrees,
    # "XGBoost": xgBoost,
    # "Random Forest": randomForest,
    # "SVM": SVM
}


for name, model in models.items():
  print(f"Training Model: {name}")
  fittedModel = model.fit(X_train,y_train)
  scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
  y_pred = fittedModel.predict(X_val)
  accuracy = accuracy_score(y_val,y_pred)
  if accuracy > bestAcc:
    bestAcc = accuracy
    bestModelName = name
    bestModel = model

  print(f"Training Accuracy for {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
  print(f"Validation Accuracy for {name} is: {accuracy}")
  print("-" * 50)

print(f"The best Model is {bestModelName} with a Validation accuracy of {bestAcc}")

import joblib


# Save the best model to a .pkl file
joblib.dump(bestModel, "best_sentiment_model.pkl")

print("Model saved as best_sentiment_model.pkl!")

joblib.dump(vectorizer, "vectorizer.pkl")      # if used
joblib.dump(le, "label_encoder.pkl")           # if used

import zipfile

files_to_zip = [
    "best_sentiment_model.pkl",
    "vectorizer.pkl",
    "label_encoder.pkl",
    "prediction1.csv"
]

with zipfile.ZipFile("model_files.zip", "w") as zipf:
    for file in files_to_zip:
        zipf.write(file)

print("All files zipped into model_files.zip")
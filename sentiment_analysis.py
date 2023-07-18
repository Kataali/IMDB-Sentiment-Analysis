import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


movies_data_frame = pd.read_csv("Datasets/IMDB Dataset.csv")
# print(movies_data_frame.head()["review"])

pd.set_option('display.max_colwidth', 150)
# convert the sentiments to integers ie. positive == 1, negative == 0
movies_data_frame["sentiment"] = (movies_data_frame["sentiment"] == 'positive').astype(int)


# see first five elements in the dataframe
# print(movies_data_frame.head())

# See how big the dataset is
# print(f"shape: {movies_data_frame.shape}")

# See the number of positive reviews against negative reviews
# print(movies_data_frame["sentiment"].value_counts())

# Check for missing data
# print("Number of reviews with null values: {}".format(movies_data_frame["review"].isnull().sum()))
# print("Number of sentiments with null values: ".format(movies_data_frame["sentiment"].isnull().sum()))


# Text preprocessing functions
# Removing html entities
def strip_html(text):
    return BeautifulSoup(text, "lxml").text


# removing punctuations
def remove_punctuations(stripped_text):
    stripped_text = re.sub(r'[^\w\s]', '', stripped_text)
    return stripped_text


# Tokenization
def tokenize_text(unpuncted_text):
    tokens = nltk.word_tokenize(unpuncted_text)
    return tokens


# Removal of stopwords
def remove_stopwords(tokenized_text):
    stop_words = stopwords.words("english")
    stop_words.append("br")
    text = [word for word in tokenized_text if word not in stop_words]
    return text


# Lemmatization
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text)


# Applying preprocessing techniques on the raw review texts


def clean_text():
    clean_sentences = []
    for index, row in movies_data_frame[:25000].iterrows():
        clean_review_sentence = ""
        review_text = row["review"]
        # review_text = strip_html(review_text)
        review_text = remove_punctuations(review_text)
        tokens = tokenize_text(review_text)
        review_text = remove_stopwords(tokens)
        for word in review_text:
            clean_review_sentence = clean_review_sentence + " " + lemmatize(word)
        # movies_data_frame.iloc[index, "review"] = clean_review_sentence
        clean_sentences.append(clean_review_sentence)
    return clean_sentences


# vectorize Corpus
def vectorize():
    sample = clean_text()
    # cleaned_text = clean()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sample)
    return vectors


vectorized = vectorize()

# Split data into testing and training data
y = movies_data_frame["sentiment"][:25000]
X_train, X_test, y_train, y_test = train_test_split(vectorized, y, test_size=0.2)

# Create and Train Model
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

# Test model
y_predict = svm_model.predict(X_test)

# Get classification report
print(classification_report(y_test, y_predict))

# # Create pipeline
# my_pipeline = Pipeline([("scale", StandardScaler()), ("model", SVC())])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask import Flask, url_for, redirect, render_template, jsonify, request


# Build the flask application
application = Flask(__name__)
application.config['SECRET_KEY'] = 'hard to guess string'


# Model loading
loaded_model = None
with open('basic_classifier.pkl', 'rb') as fid:
	loaded_model = pickle.load(fid)


vectorizer = None
with open('count_vectorizer.pkl', 'rb') as vd:
	vectorizer = pickle.load(vd)
    

# Welcome page
@application.route("/")
def welcome():
    return 'welcome'


# predict whether the input is a fack news by the 'sentence' param sent by the get request
@application.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    prediction = loaded_model.predict(vectorizer.transform([sentence]))[0]
    return prediction


if __name__ == '__main__':
    application.debug = True
    application.run()

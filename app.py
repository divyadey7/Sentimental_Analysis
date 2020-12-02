from flask import Flask, render_template, flash, request, url_for
import numpy as np
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from sklearn.linear_model import LogisticRegression
from numpy import array
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer

IMAGE_FOLDER=os.path.join('static','img_pool')
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

def init():
    global pipeline
    pipeline = load("analysis_pipeline.joblib")

@app.route('/', methods=['GET','POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods=['POST','GET'])
def sent_anly_prediction():
    if request.method=='POST':
        text=request.form['text']
        sentiment=''

        my_prediction=pipeline.predict([text])
        if my_prediction=='negative':
            sentiment='NEGATIVE'
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'sad_emoji.png')
        else:
            sentiment='POSITIVE'
            img_filename=os.path.join(app.config['UPLOAD_FOLDER'], 'smile_emoji.png')
    return render_template('home.html',text=text, sentiment=sentiment, image=img_filename)

if __name__ == '__main__':
    init()
    app.debug=True
    app.run()

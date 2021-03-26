# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:09:14 2021

@author: keerthi
"""

from flask import Flask, request, render_template
from keras.models import load_model
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import os
    
app = Flask(__name__, template_folder="template")
filename = os.getcwd()
print(filename)
p = 'D:/internship/donorchoose/test.h5'
#model_path = os.path.join(filename,p)
model=load_model(p) 
#dnr=load_model('') 

@app.route("/", methods=["GET"])
def home():
    return render_template('dnrchoose.html')

# set hyper parameters for title
vocab_size = 69191
embed_dim = 64
max_length = 200
trunc_type = 'pre'
padding_type = 'pre'
oov_tok = '<OOV>'

nltk.download('stopwords')
nltk.download('wordnet')

sw = stopwords.words('english')
lemma = WordNetLemmatizer()

def preprocess_data(text):  
    text = text
    text=re.sub('[^a-zA-Z0-9]',' ',text)                        # Remove special characters(punctuations) and numbers 
    text=text.lower()                                           # Convert to lower case
    text=text.split()                                           # Tokenization
    text = [word for word in text if word not in sw]            # Removing stopwords
    text = [lemma.lemmatize(word=w,pos='v') for w in text]      # lemmatization
    text = [k for k in text if len(k)>2]                        # Remove words with length < 2
    text = ' '.join(text)
    
  
    ohe = [one_hot(word, vocab_size) for word in text]
    padded = pad_sequences(ohe, padding=padding_type, truncating=trunc_type)
    fd = (pd.DataFrame(padded)).transpose()
    return fd

@app.route("/prediction", methods=["POST"])
def deploy():
    label = {0:'Project is not approved', 1:'Project is approved'}
    x_col = ['project_grade_category','teacher_number_of_previously_posted_projects', 'project_title',
             'project_subject_subcategories','project_essay_1','project_essay_2']
    
    # data from end user
    data = [[x for x in request.form.values()]]
    d = pd.DataFrame(data,columns=x_col)
    
    t = preprocess_data(d['project_title'][0])
    s = preprocess_data(d['project_subject_subcategories'][0])
    e1 = preprocess_data(d['project_essay_1'][0])
    e2 = preprocess_data(d['project_essay_2'][0])

    dataset = (pd.DataFrame([d['project_grade_category'],d['teacher_number_of_previously_posted_projects']])).transpose()    
    dataset = pd.concat([dataset,t,s,e1,e2],axis=1)
    dataset = np.asarray(dataset).astype(np.int)
    predict = model.predict_classes(dataset)[0]
    text = label[predict[0]]
    return render_template('dnrchoose.html',prediction = text)
if __name__ == '__main__':
    app.run(debug=True)

    
        
        
        
        
        
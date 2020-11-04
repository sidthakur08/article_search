import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from ast import literal_eval

import string
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import KeyedVectors

print("Downloading the wordnet from nltk...")
import nltk
nltk.download('wordnet')

data = pd.read_csv('data/data.csv')
data = data.dropna()
stopwords_eng = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_backslash(text):
    return text.replace('\n',' ').replace('\r',' ').replace('\xa0',' ')

def process_text(text):
    text = text.lower()

    punc_list = '!"#$%()*+,-./:;<=>?@^_{|}~'
    t = str.maketrans(dict.fromkeys(punc_list," "))
    text = text.translate(t)

    t = str.maketrans(dict.fromkeys("'`",""))
    text = text.translate(t)

    tokens = regexp_tokenize(text,pattern='\s+',gaps=True)
    cleaned_tokens = []

    for t in tokens:
        if t not in stopwords_eng:
            l = lemmatizer.lemmatize(t)
            cleaned_tokens.append(l)

    return cleaned_tokens

def get_vec(word):
    try:
        return model[word]
    except:
        return np.zeros(300)

print("Loading the word2vec model...")
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True,limit=10**6)
data_dict = []
print("Tokenizing and Getting the sentence vector...")
for i in tqdm(range(data.shape[0])):
    url = data.iloc[i]['url']
    headline = remove_backslash(data.iloc[i]['headline'])
    tokens = process_text(headline)
    vector = sum([get_vec(t) for t in tokens]).tolist()
    data_dict.append({
        'url':url,
        'headline':headline,
        'tokens':tokens,
        'sentence_vector':vector
    })

df = pd.DataFrame(data_dict)
print("Finding zero sentence vectors...")
for i in tqdm(range(df.shape[0])):
    try:
        if (df.iloc[i]['sentence_vector'] == np.zeros(300)).all():
            df = df.drop([i],axis=0)
    except Exception as e:
        print(e)

df = df.reset_index(drop=True)

print("Saving the data...")
df.to_csv('data/new_data.csv',index=False)
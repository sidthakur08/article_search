import numpy as np
import glob
import json
from tqdm import tqdm

import string
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import KeyedVectors

print("Downloading the wordnet from nltk...")
import nltk
nltk.download('wordnet')

file = 1
paths = glob.glob(f'./articles_data/{file}/*.json')

article_data = []
print("Adding article data...")
for path in tqdm(paths):
    with open(path) as f:
        article_data.append(json.load(f))

stopwords_eng = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def process_text(text):
    text = text.replace("\n"," ").replace("\r"," ")
    
    t = str.maketrans(dict.fromkeys("'`",""))
    text = text.translate(t)
    
    tokens = regexp_tokenize(text,pattern='\s+',gaps=True)
    cleaned_tokens = []
    
    for t in tokens:
        if t not in stopwords_eng and t not in string.punctuation:
            l = lemmatizer.lemmatize(t)
            cleaned_tokens.append(l)
    
    return cleaned_tokens

def get_vec(word):
    try:
        return model[word]
    except:
        return np.zeros(300)

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True,limit=10**6)
data = []
print("Tokenizing and Getting the sentence vector...")
for i in tqdm(range(len(article_data))):
    full_t = article_data[i]['thread']['section_title']+' '+article_data[i]['thread']['title_full']
    url = article_data[i]['thread']['url']
    tokens = process_text(full_t)
    sent_vector = sum([get_vec(t) for t in tokens]).tolist()

    data.append({
        'full_title':full_t,
        'url':url,
        'title_tokens':tokens,
        'sentence_vector':sent_vector
    })

print("Saving the sentence vectors...")
with open(f"data_{file}.json","w") as f:
    json.dump(sent_vector,f)
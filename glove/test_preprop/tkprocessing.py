import os
import nltk
import pickle
import re
from bs4 import BeautifulSoup

dataDir = "data/"

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

for f in os.listdir(dataDir):
    print("Now Processing: ", f)
    with open(os.path.join(dataDir, f), 'r') as review:
        #Remove HTML
        review_text = BeautifulSoup(review, "lxml").get_text()

        #Remove punctuation (falcultative)
        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        #Standard English Tokenizer
        tokens = nltk.word_tokenize(review_text.lower())
        sentences+=tokens

corpus = " ".join(sentences)
with open("output.txt", "w") as text_file:
    text_file.write(corpus)

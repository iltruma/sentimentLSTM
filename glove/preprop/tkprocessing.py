import os
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

corpus = ""
dataDir = ["../../data/aclImdb/train/pos", "../../data/aclImdb/train/neg", "../../data/aclImdb/train/unsup"]
#dataDir = ["../../data/aclImdb/train/unsup"]
#dataDir = ["input"]

def new_tokenizer(review, punct, stop):
    '''CC
    Tokenize a review:
        - Remove HTML
        - Remove Remove punctuation (optional)
        - Tokenize and lowercase the words
        - Remove English stopwords (optional)
    '''
    review_text = BeautifulSoup(review, "lxml").get_text()

    if punct: review_text = re.sub("[^a-zA-Z]"," ", review_text)

    tokens = nltk.word_tokenize(review_text.lower())

    if stop:
        stop = set(stopwords.words('english'))
        tokens = [i for i in tokens if i not in stop]

    return tokens

#Tokenize every file of dataDir and merge them together
for dir in dataDir:
    print("Now processing folder: " + dataDir + "\n")

    for f in os.listdir(dataDir):
        with open(os.path.join(dataDir, f), 'r') as review:
            review_tkn = new_tokenizer(review.read(), punct=True, stop=True)
            corpus += " ".join(review_tkn) + "\n"

with open("output.txt", "w") as text_file:
    text_file.write(corpus)
    print("Finished :D")

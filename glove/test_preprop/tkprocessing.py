import os
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--punct", help="remove punctuation", action="store_true")
parser.add_argument("--stop", help="remove stopwords", action="store_true")
args = parser.parse_args()

args = parser.parse_args()

dataDir = "input/"
corpus=""

for f in os.listdir(dataDir):
    print("Now Processing: ", f)
    with open(os.path.join(dataDir, f), 'r') as review:
        #Remove HTML
        review_text = BeautifulSoup(review, "lxml").get_text()

        #Remove punctuation (falcultative)
        if args.punct:
            review_text = re.sub("[^a-zA-Z]"," ", review_text)

        #Standard English Tokenizer
        tokens = nltk.word_tokenize(review_text.lower())

        #Remove english stopwords (falcultative)
        if args.stop:
            stop = set(stopwords.words('english'))
            tokens = [i for i in tokens if i not in stop]

        corpus += " ".join(tokens) + "\n"


with open("output.txt", "w") as text_file:
    text_file.write(corpus)

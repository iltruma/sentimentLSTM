import os
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

#dataDir = "../../data/aclImdb/train/unsup"
dataDir = "input"
corpus = ""
count = 0

print("Now processing folder: " + dataDir + "\n")

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
for f in os.listdir(dataDir):
    with open(os.path.join(dataDir, f), 'r') as review:

        review_tkn = new_tokenizer(review, punct=True, stop=True)
        corpus += " ".join(review_tkn) + "\n"

        count+=1
        progress = count/len(os.listdir(dataDir))*100
        if (progress)%5 == 0:
            print(progress,"% percent complete")


with open("output.txt", "w") as text_file:
    text_file.write(corpus)
    print("\nFinished :D")

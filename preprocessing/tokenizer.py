import os
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import argparse

corpus = ""
dataDirs = ["../../data/aclImdb/train/pos", "../../data/aclImdb/train/neg", "../../data/aclImdb/train/unsup"]
# dataDirs = ["../../data/aclImdb/train/pos"]
# dataDirs = ["input"]

print("Tokenizer started")


def tokenize(review, punct=True, stop=False):
    """CC
    Tokenize a review:
        - Remove HTML
        - Remove Remove punctuation (optional)
        - Tokenize and lowercase the words
        - Remove English stopwords (optional)
    """
    review_text = BeautifulSoup(review, "html5lib").get_text()

    if punct: review_text = re.sub("[^a-zA-Z]", " ", review_text)

    tokens = nltk.word_tokenize(review_text.lower())

    if stop:
        stop = set(stopwords.words('english'))
        tokens = [i for i in tokens if i not in stop]

    return tokens


if __name__ == '__main__':
    # Tokenize every file of dataDirs and merge them together
    parser = argparse.ArgumentParser(description='Tokenizer for Glove')
    parser.add_argument('--punct', action='store_true', help='remove punctuation from corpus')
    parser.add_argument('--stop', action='store_true', help='remove stopwords from corpus')
    args = parser.parse_args()

    for dir in dataDirs:
        print("\tNow processing folder: " + dir)

        for f in os.listdir(dir):
            with open(os.path.join(dir, f), 'r') as review:
                review_tkn = tokenize(review.read(), args.punct, args.stop)
                corpus += " ".join(review_tkn) + "\n"
                review.close()

    with open(
            "../../data/train_glove{p}{s}".format(p="_nopunct" if args.punct else "", s="_nostop" if args.stop else ""),
            "w") as text_file:
        text_file.write(corpus)
        text_file.close()

    print("Finished :D")

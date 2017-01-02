import os
import nltk
import pickle
import urllib
import numpy as np
from multiprocessing import Process, Lock

dataDir = "../data/"
dirs = [dataDir + "aclImdb/train/unsup"]

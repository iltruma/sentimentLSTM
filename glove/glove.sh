#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS=corpus.txt
DATADIR=../data/
VERBOSE=2
EVALUATE=false
GRAPH=false
BINARY=2
NUM_THREADS=8

VOCAB_MIN_COUNT=5
VECTOR_SIZE=200
MAX_ITER=30
LEARNINGRATE=0.05
WINDOW_SIZE=5
X_MAX=100


#Create Vocabulary with no limit of dimension
build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $DATADIR$CORPUS > out/vocab.txt

#Create Co-occurrence Matrix
build/cooccur -memory 4.0 -vocab-file out/vocab.txt -verbose $VERBOSE -symmetric 1 -window-size $WINDOW_SIZE < $DATADIR$CORPUS > out/cooccurrence.bin

#Shuffle Co-occurrence Matrix
build/shuffle -memory 4.0 -verbose $VERBOSE < out/cooccurrence.bin> out/cooccurrence.shuf.bin

#Run Glove Model
build/glove -save-file out/vectors -threads $NUM_THREADS -input-file out/cooccurrence.shuf.bin -iter $MAX_ITER -vector-size $VECTOR_SIZE -vocab-file out/vocab.txt -x-max $X_MAX -alpha 0.75 -eta $LEARNINGRATE -verbose $VERBOSE -binary $BINARY

if $EVALUATE
  then
    echo "EVALUTING THE MODEL"
    python3 eval/python/evaluate.py
fi
if $GRAPH
  then
    echo "I'm building the graph of embeddings..."
    python3 eval/python/graph.py
fi

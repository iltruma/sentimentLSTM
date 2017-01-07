#!/bin/bash
CORPUS=train_glove_nopunct
DATADIR=../data/
VERBOSE=2
BINARY=0
NUM_THREADS=8

VOCAB_MIN_COUNT=5
MAX_ITER=30
LEARNINGRATE=0.05
X_MAX=100


for X in 50 100 150 200 250 300
do
  for Y in 5 10 15 20
  do
    echo "=============DIMENSIONS============" $X
    echo "============WINDOW_SIZE============" $Y
    build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $DATADIR$CORPUS > out/vocab.txt

    #Create Co-occurrence Matrix
    build/cooccur -memory 4.0 -vocab-file out/vocab.txt -verbose $VERBOSE -symmetric 1 -window-size $Y < $DATADIR$CORPUS > out/cooccurrence.bin

    #Shuffle Co-occurrence Matrix
    build/shuffle -memory 4.0 -verbose $VERBOSE < out/cooccurrence.bin> out/cooccurrence.shuf.bin

    #Run Glove Model
    build/glove -save-file out/vectors -threads $NUM_THREADS -input-file out/cooccurrence.shuf.bin -iter $MAX_ITER -vector-size $X -vocab-file out/vocab.txt -x-max $X_MAX -alpha 0.75 -eta $LEARNINGRATE -verbose $VERBOSE -binary $BINARY

    echo "EVALUTING THE MODEL"
    python3 eval/python/evaluate.py
  done
done

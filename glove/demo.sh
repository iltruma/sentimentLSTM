#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

DATADIR=../data
OUTDIR=out
CORPUS=$DATADIR/train-unsup
VOCAB_FILE=$OUTDIR/vocab.txt
COOCCURRENCE_FILE=$OUTDIR/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$OUTDIR/cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=$OUTDIR/vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=200
MAX_ITER=1
LEARNINGRATE=0.05
WINDOW_SIZE=10
BINARY=0
NUM_THREADS=8
X_MAX=100
EVALUATE=true
GRAPH=true

#Create Vocabulary
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

#Create Co-occurrence Matrix
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

#Shuffle Co-occurrence Matrix
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

#Run Glove Model
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -iter $MAX_ITER -vector-size $VECTOR_SIZE -vocab-file $VOCAB_FILE -eta $LEARNINGRATE -binary 0 -verbose 2

if $EVALUATE
  then
    echo ""
    python3 eval/python/evaluate.py
fi
if $GRAPH
  then
    echo "I'm building the graph of embeddings..."
    python3 eval/python/graph.py
fi

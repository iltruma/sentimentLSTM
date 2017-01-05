import subprocess
vocab = "out/vocab.txt"

#subprocess.call(["build/cooccur", "-memory 4.0", "-vocab-file " + vocab, "-verbose 2", "-symmetric 1", "-window-size 10", "<../data/corpus.txt>", "out/cooccurrence.bin"])
subprocess.call("build/cooccur -memory 4.0 -vocab-file " + vocab + " -verbose 2 -symmetric 1 -window-size 10 <../data/corpus.txt > out/cooccurrence.bin", shell=True)

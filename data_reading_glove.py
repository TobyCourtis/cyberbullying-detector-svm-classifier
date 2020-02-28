import pandas as pd
import pickle
import numpy

index_map = dict()
vectors = numpy.empty((1917494,300), dtype=numpy.float32)
with open("/Users/tobycourtis/Desktop/GloVe-1.2/_Extra/glove.42B.300d.txt","r") as f:
    #print(len(f.readlines()))
    for i, line in enumerate(f):
        word, *rest = line.split()
        index_map[word] = i
        vectors[i, :] = numpy.asarray(rest, dtype=numpy.float32)



output = open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map2.pkl', 'wb')
pickle.dump(index_map, output)
output.close()

output2 = open('/Users/tobycourtis/Desktop/Data/final_form_data/vectors2.pkl', 'wb')
pickle.dump(vectors, output2)
output2.close()



print("Saved Index Map and Vectors -- exited")

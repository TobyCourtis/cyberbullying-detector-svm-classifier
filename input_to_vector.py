import pandas as pd
import pickle
import numpy

with open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)

vectors2 = numpy.load("/Users/tobycourtis/Desktop/Data/final_form_data/vectors-float32.npz")
vectors = vectors2['arr_0'][:]

while True:
    textInput = input("Input a sentence for vectorisation: ")

    text = textInput.split()

    sentence_vector = numpy.zeros(300, dtype=numpy.float32)
    for word in text:
        try:
            sentence_vector += vectors[index_map[word]]
        except:
            pass
    sentence_vector /= len(text)

    print(list(sentence_vector))
    print("Length of vector: ", len(sentence_vector))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import json

print("loading vectors...")
with open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)

vectors2 = np.load("/Users/tobycourtis/Desktop/Data/final_form_data/vectors-float32.npz")
vectors = vectors2['arr_0'][:]
print("Done loading vectors... ")


with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/twitter_data_json80.json') as json_data:
    d = json.load(json_data)

# classiifer file name here
filename = "classifier_wiki_data_json80.sav"

loaded_model = pickle.load(open(filename, 'rb'))

data_input = []
data_input_labels = []
def sentence(toUse):
    text = toUse.split()
    sentence_vector = np.zeros(300, dtype=np.float32)
    for word in text:
        # here need to do a try so that if the word doesn't exist it still works
        # also need to round to 5sf (or 4dp)
        try:
            sentence_vector += vectors[index_map[word]]
        except:
            pass
    # average found here
    sentence_vector /= len(text)

    # print(list(sentence_vector))
    # print("Length of vector: ", len(sentence_vector))
    return list(sentence_vector)

for i in d:
    splitUp = i['text'].split()
    if len(splitUp) > 0:
        data_input.append(sentence(i['text']))
        data_input_labels.append(i['label'])
    else:
        print("No text given with label, error ")

result = loaded_model.score(np.array(data_input),data_input_labels,sample_weight=None)
print(result)
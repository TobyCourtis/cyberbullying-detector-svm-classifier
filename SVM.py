import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import json

# the data set used
dataSet = "combined_data_json80"

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/'+ dataSet + '.json') as json_data:
    data = json.load(json_data)

# reading in the vector mapping section -- later used in classifying
print("loading vectors...")
with open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)

vectors2 = np.load("/Users/tobycourtis/Desktop/Data/final_form_data/vectors-float32.npz")
vectors = vectors2['arr_0'][:]
print("Done loading vectors... ")
# input here need to take the text from the training data being used
def sentence(toUse):
    text = toUse.split()
    sentence_vector = np.zeros(300, dtype=np.float32)
    for word in text:
        try:
            sentence_vector += vectors[index_map[word]]
        except:
            pass
    # average found here
    sentence_vector /= len(text)
    return list(sentence_vector)



# read in data using read in -> 'data'
print("Collecting data...")

data_input = []
data_input_labels = []
for i in data:
    splitUp = i['text'].split()
    if len(splitUp) > 0:
        data_input.append(sentence(i['text']))
        data_input_labels.append(i['label'])
    else:
        print("No text given with label, error passed ")


print("Finished collecting data...")

#######
#######
####### - classifier
#######
#######

classifier = svm.SVC(kernel="linear", C = 1.0)

print("Fitting Classifier...")
model = classifier.fit(np.array(data_input),data_input_labels,sample_weight=None)
print("Finished fitting Classifier...")

# saving the classifier
print("Saving Classifier...")
filename = 'classifier_' + dataSet + '_rbf.sav'
pickle.dump(model, open(filename, 'wb'))
print("Finished saving Classifier...")


print("\nFinished all and saved classifier\n")

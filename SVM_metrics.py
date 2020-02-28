import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import json
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, cohen_kappa_score

# changing the data set used below
dataSet = "combined_data_json80"
#dataSet = "combined_data_json80"

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/'+ dataSet + '.json') as json_data:
    data = json.load(json_data)


testSet = "combined_data_json20"
#dataSet = "combined_data_json80"

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/'+ dataSet + '.json') as json_data:
    data2 = json.load(json_data)

# reading in the vector mapping section -- later used to classify
print("loading vectors...")
with open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)

vectors2 = np.load("/Users/tobycourtis/Desktop/Data/final_form_data/vectors-float32.npz")
vectors = vectors2['arr_0'][:]
print("Done loading vectors... ")
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

# read in data
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

print("Collecting data No.2...")
data_input2 = []
data_input_labels2 = []
for i in data2:
    splitUp = i['text'].split()
    if len(splitUp) > 0:
        data_input2.append(sentence(i['text']))
        data_input_labels2.append(i['label'])
    else:
        print("No text given with label, error passed ")


print("Finished collecting data2...")


# classifier file name here
filename = "classifier_combined_data_json80.sav"

classifier = pickle.load(open(filename, 'rb'))

y_true_vals = data_input_labels2

y_pred_vals = []
for i in data_input2:
    test = np.reshape(i,(1,-1))
    y_pred_vals.append(classifier.predict(test)[0])

print("Finished and finding scores:")
print("f1 macro", f1_score(y_true_vals, y_pred_vals,average="macro"))
print("precision macro", precision_score(y_true_vals, y_pred_vals, average="macro"))
print("recall macro", recall_score(y_true_vals, y_pred_vals, average="macro"))
print("accuracy", accuracy_score(y_true_vals, y_pred_vals))
print("kappa123", cohen_kappa_score(y_true_vals,y_pred_vals))


print("\nFinished all\n")

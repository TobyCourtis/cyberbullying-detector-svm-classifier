import pandas as pd
import pickle
import json
import numpy

def load_obj(name):
    with open('/Users/tobycourtis/Desktop/Data/data_Sweta20/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

data = load_obj("wiki_data")

# method for twitter data
for i in range(len(data)):
    if i == 0:
        print(data[i])
    if (data[i]['label'] == 'none'):
        data[i]['label'] = 0
    else:
        data[i]['label'] = 1



'''
method for wiki data:



newData = []

for i in range(len(data)):
    temp = {'label':"",'text':""}
    temp['label'] = data[i]['label']
    temp['text'] = data[i]['text']
    newData.append(temp)


for i in range(len(newData)):
    if (newData[i]['label'] != 0):
        print(newData[i])

with open('/Users/tobycourtis/Desktop/Data/data_Sweta20/twitter_data_json32.json', 'w') as outfile:
    json.dump(data, outfile)
'''

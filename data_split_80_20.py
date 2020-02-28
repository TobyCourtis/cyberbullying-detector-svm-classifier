import pickle
import json

toSplit = "wiki_data_json"

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/'+ toSplit + '.json') as json_data:
    d = json.load(json_data)
# 80% train data
data1 = []
# 20% test data
data2 = []

print("Length before: ",len(d))
# 0 = not bullying
# length of 80%
count_label0 = 81819
count = 0
# 1 = bullying
# length of 20%
count_label1 = 10872


for i in d:
    if i['label'] == 0:
        if count_label0 > 0:
            data1.append(i)
        else:
            data2.append(i)
        count_label0 -= 1
    else:
        if count_label1 > 0:
            data1.append(i)
        else:
            data2.append(i)
        count_label1 -= 1

print("Finished")
print("data1 length: ",len(data1))
print("data2 length: ",len(data2))

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/' + toSplit + "80" + '.json', 'w') as outfile:
    json.dump(data1, outfile)

with open('/Users/tobycourtis/Desktop/Data/final_form_data/data/' + toSplit + "20" + '.json', 'w') as outfile:
    json.dump(data2, outfile)

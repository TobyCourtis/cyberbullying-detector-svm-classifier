from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import ndjson
import nltk
nltk.download('punkt')


corpus = [
'All my cats in a row',
'When my cat sits down, she looks like a Furby toy!',
'The cat from outer space',
'Sunshine loves to sit like this for some reason.'
]

# method here is generic method of Bag of Words given by scikit.

vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense() )
print(vectorizer.vocabulary_ )
print("")
print("--------------")
print("")

# bigram method

bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1)
analysis = bigram_vectorizer.build_analyzer()
print(analysis('Bi-grams are cool!'))
print(bigram_vectorizer.fit_transform(corpus).todense())


# here is the method of opening the data in ndjson format - can convert to json if needed.
# opening of the ndjson - certain lines can be selected

with open('../Data/sampled_post-comments_vine.json') as f:
    data = ndjson.load(f)

print("")
print("--------------")
print("")
file = open("testFile.txt","w")
for j in range(len(data)):
    file.write("" + data[j]['commentText'] + "\n")
file.close()
print("Finsihed Writing")

# nltk

print("")
print("--------------")
print("")

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print(tokens)


print("")
print("--------------")
print("")

# reading in csv files here - byte errot with labeled_10plus_to_40_full.csv

#data2 = pd.read_csv('tobycourtis/Desktop/Data/labeled_10plus_to_40_full.csv')
#print(data2[0])

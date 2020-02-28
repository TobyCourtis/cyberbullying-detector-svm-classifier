# cyberbullying-detector-svm-classifier
SVM (classifier) component of the 'Cyberbullying Detector' used to produce classifer '.sav' file for usage in cyberbullying-detector-server

<h3>All files were used in the creation of the final SVM classifier used in my 'Cyberbullying Detector'</h3>
<ul>
<li><b>Word Representation</b>: <a href="https://nlp.stanford.edu/projects/glove/">GloVe</a> Vectors (42B tokens, 300d vectors)</li>
  <ul>
    <li>Training corpus for GloVe Model - first 100M characters of Wikipedia</li>
  </ul>
</ul>
<ul>
<li><b>Word to GloVe Vector</b>: Uses two files - 'index_map' (dictionary) and 'vectors' (array):</li>
  <ul>
    <li>index_map returns the index of a given word in the vectors array</li>
    <li>The 300 dimension vectors are found through the index_map index - allowing simple lookup when training classifier</li>
  </ul>
</ul>
<ul>
<li><b>Training Data</b>:</li>
  <ul>
  <li>Pre-Classified: Twitter (~16k posts) and Wikipedia Forum (~100k posts)</li>
  <li>Shuffled and split into 80% training and 20% test (split by bullying/non-bullying)</li>
  </ul>
</ul>
<ul>
<li><b>Classifer</b>: SVM using scikit-learn library</li>
  <ul>
  <li>Trained using aquired 80% training data and GloVe vector lookup and input</li>
  <li>Training parameters varied whilst prototyping with accuracy tested against 20% testing data</li>
  </ul>
</ul>

# cyberbullying-detector-svm-classifier
SVM (classifier) component of the 'Cyberbullying Detector' used to produce classifer '.sav' file for usage in cyberbullying-detector-server
<br></br>
My research paper titled ‘Towards Automated Blocking of Cyberbullying in Online Social Networks’, outlining the background and approach to producing my cyberbullying detector can be viewed [here](https://www.tobycourtis.com/content/Towards-Automated-Blocking-of-Cyberbullying-in-OSNs.pdf).
<br></br>
Experiment by inputting text to the classifier through the Google Chrome extension user interface [here](https://www.tobycourtis.com/content/cyberbullying_detector).
<br></br>
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
  <li>Download a few of the best performing trained classifiers <a href="https://drive.google.com/open?id=1_RR5WVZt3rrFuTVHwXuuruheb48BZcE_">here</a></li>
  <li>The best performing classifier is included in the repository within /classifiers</li>
  </ul>
</ul>

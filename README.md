# Recognition-of-Handwritten-Alphabet-tensorflow-
This is a simply project for recognition of Handwritten Alphabet using tensorflow.The objective is to identify each of a large number of black-and-white rectangular
pixel displays as one of the 26 capital letters in the English alphabet. The character images were based on 20 different fonts and each letter within these 20
fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments
and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. The problem is solved by training the multi-layer neural network followed
by a logistic regressor then make the classification.

The character of dataset
The data set contains a labelled training data set (20000 lines), ‘letter recognition training data set.csv’, and a non-labelled testing data set
(3000 lines), ‘letter recognition testing data set.csv’. The task is to recognize these characters in the test set according to its input features and
grading will be partially based on classification accuracy.

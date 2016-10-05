from __future__ import division
import sys
import csv
import argparse
import util
from collections import defaultdict

# import util

import ast
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tokenizer import Tokenizer
import matplotlib.pyplot as plt
import porter_stemmer
import operator

def main():
	##### DO NOT MODIFY THESE OPTIONS ##########################
	parser = argparse.ArgumentParser()
	parser.add_argument('-training_1', required=True, help='Path to training data')
	parser.add_argument('-training_2', required=True, help='Path to training data')

	# Training data:

	# Training_1 = (year , team), rank
	# Training_2 = (year, team), [{average salary: x}, {standard dev salary: y}, {team batting average: z} ...]

	parser.add_argument('-test', help='Path to test data')

	# Test data: 2014 data

	parser.add_argument('-c', '--classifier', default='nb', help='nb | log | svm')
	parser.add_argument('-top', type=int, help='Number of top features to show')

	opts = parser.parse_args()
	############################################################

	##### BUILD TRAINING SET ###################################

	# Initialize CountVectorizer
	# You will need to implement functions in tokenizer.py

	tokenizer = Tokenizer()
	vectorizer = DictVectorizer()

	# Load training text and training labels

	# Get training features using vectorizer
	
	# Transform training labels to numpy array (numpy.array)

	training_data_1 = open(opts.training_1, "rb")
	reader_1 = csv.reader(training_data_1)

	training_data_2 = open(opts.training_2, "rb")
	reader_2 = csv.reader(training_data_2)


	# Training_1 = (year , team), rank
	# Training_2 = (year, team), [average salary: standard dev salary: team batting average: team pitching ERA ...]

	all_teams_features = []
	all_standings = []

	for item in reader_1: # You are comparing all features (i.e. batting average, salary) with team standings
		year = item[0][0]
		team = item[0][1]
		rank = int(item[1])
		all_standings.append(rank)
	for item in reader_2:
		year = item[0][0]
		team = item[0][1]
		features = ast.literal_eval(item[1])
		all_teams_features.append(features)

	features_vector = vectorizer.fit_transform(all_teams_features)
	standings_vector = numpy.array(all_standings)
	
	############################################################

	##### TRAIN THE MODEL ######################################
	# Initialize the corresponding type of the classifier and train it (using 'fit')
	if opts.classifier == 'nb':
		print "nb"
		classifier = linear_model.LinearRegression()
		classifier.fit(features_vector, standings_vector)
		print "coef: ", classifier.coef_
		
	else:
		raise Exception('Unrecognized classifier!')
	############################################################


	###### VALIDATE THE MODEL ##################################
	# Print training mean accuracy using 'score'
	
	# Perform 10 fold cross validation (cross_validation.cross_val_score) with scoring='accuracy'
	# and print the mean score and std deviation
	
	############################################################

	training_accuracy = classifier.score(features_vector, standings_vector)
	print "Mean accuracy: ", training_accuracy


def combinations(iterable, r):
	# http://docs.python.org/2/library/itertools.html#itertools.combinations
	# combinations('ABCD', 2) --> AB AC AD BC BD CD
	# combinations(range(4), 3) --> 012 013 023 123
	pool = tuple(iterable)
	n = len(pool)
	if r > n:
		return
	indices = range(r)
	yield tuple(pool[i] for i in indices)
	while True:
		for i in reversed(range(r)):
			if indices[i] != i + n - r:
				break
		else:
			return
		indices[i] += 1
		for j in range(i+1, r):
			indices[j] = indices[j-1] + 1
		yield tuple(pool[i] for i in indices)

def correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy):
	# http://en.wikipedia.org/wiki/Correlation_and_dependence
	numerator = n * sum_xy - sum_x * sum_y
	denominator = math.sqrt(n * sum_xx - sum_x * sum_x) * math.sqrt(n * sum_yy - sum_y * sum_y)
	if denominator == 0:
		return 0.0
	return numerator / denominator

def regularized_correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, virtual_count, prior_correlation):
	unregularized_correlation_value = correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
	weight = n / (n + virtual_count)
	return weight * unregularized_correlation_value + (1 - weight) * prior_correlation

def cosine_similarity(sum_xx, sum_yy, sum_xy):
	# http://en.wikipedia.org/wiki/Cosine_similarity
	numerator = sum_xy
	denominator = (math.sqrt(sum_xx) * math.sqrt(sum_yy))
	if denominator == 0:
		return 0.0
	return numerator / denominator

def jaccard_similarity(n_common, n1, n2):
	# http://en.wikipedia.org/wiki/Jaccard_index
	numerator = n_common
	denominator = n1 + n2 - n_common
	if denominator == 0:
		return 0.0
	return numerator / denominator

 		
if __name__ == '__main__':
	main()

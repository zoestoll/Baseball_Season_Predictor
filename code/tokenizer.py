from porter_stemmer import PorterStemmer
import re
import string

class Tokenizer(object):
	def __init__(self):
		self.stemmer = PorterStemmer()

	def __call__(self, tweet):
		# TODO: This function takes in a single tweet (just the text part)
		# then it will process/clean the tweet and return a list of tokens (words).
		# For example, if tweet was 'I eat', the function returns ['i', 'eat']

		# You will not need to call this function explictly. 
		# Once you initialize your vectorizer with this tokenizer, 
		# then 'vectorizer.fit_transform()' will implictly call this function to 
		# extract features from the training set, which is a list of tweet texts.
		# So once you call 'fit_transform()', the '__call__' function will be applied 
		# on each tweet text in the training set (a list of tweet texts),
		tweet.lower(); # 1. Lowercase all letters
		for i in string.punctuation:
			tweet = tweet.replace(i, " ")
		words = tweet.split()

		result = []
		for word in words:
			if word[0] == "@": # 7. Removing user references
				word = "AT_USER"
			if word[0] == "\#": # 5. Removing hashtags
				word[0] = word[0].replace("\#", "")	
			if word[0].isalpha(): # Ignoring words that don't start with an alphabet letter
				if word.startswith("www.") or word.startswith("https://") or word.startswith("http://"):
					word = "URL"
				word = PorterStemmer().stem(word, 0,len(word)-1) # 2. Applying stemming
				word = re.sub(r'([a-z])\1+', r'\1\1', word)
				result.append(word)
				
		return result

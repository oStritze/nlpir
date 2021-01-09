import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string
import glob
import os
import math
from typing import List


def setup_nltk():
	nltk.download('stopwords')
	nltk.download('punkt')


def tfidf_per_doc(president_dir: str):
	data_path = glob.glob("../data/{}/".format(president_dir))
	tokens_docs = _read_all_text_files(data_path[0])
	word_count_docs, dfs = _count_words_per_doc(tokens_docs)

	n_docs = len(word_count_docs)
	tfidf_docs = []
	for i, word_count in enumerate(word_count_docs):
		doc_size = len(tokens_docs[i])
		tfidf = {}
		for token, occ in word_count.items():
			tf = occ / doc_size
			idf = math.log(n_docs / dfs[token])
			tfidf[token] = tf * idf
		tfidf_docs.append(tfidf)
	return tfidf_docs


def _read_all_text_files(data_path: str):
	"""
	:return: a list of lists: for each text document it contains the list of tokens of the text document
	"""
	file_tokens = []
	for filename in os.listdir(data_path):
		input_file = glob.glob(data_path + filename)
		text = open(input_file[0], "r", encoding="utf8").read()
		file_tokens.append(_preprocess_text(text))
	return file_tokens


def _preprocess_text(text: str):
	"""
	:return: a list of tokens of the text, stopwords and punctuation removed, lowercased
	"""
	stop_words = set(stopwords.words('english'))
	punctuations = list(string.punctuation)
	return [token.lower() for token in word_tokenize(text) if token not in stop_words and token not in punctuations]


#
def _count_words_per_doc(tokens_docs: List[List]):
	"""
	:param tokens_docs: list of tokens per document
	:return: list of dictionaries with counts of each word of the vocabulary per document
	"""
	vocab = set(sum(tokens_docs, []))
	word_counts_docs = []
	dfs = {token: 0 for token in vocab}
	for tokens in tokens_docs:
		word_count = {token: 0 for token in vocab}
		for token in tokens:
			word_count[token] += 1
		for token in set(tokens):
			dfs[token] += 1
		word_counts_docs.append(word_count)
	return word_counts_docs, dfs


def _create_tf_dict(number_of_words: dict):
	"""
	Create a dictionary with the normalized tf scores per token, the tokens of the vocabulary are the keys, the tf scores are the values
	"""
	doc_size = sum([occ for _, occ in number_of_words.items()])
	return {token: occ / doc_size for token, occ in number_of_words.items()}

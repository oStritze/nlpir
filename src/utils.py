import nltk
from nltk.tokenize import word_tokenize

import os
import math
from typing import List


def setup_nltk():
	nltk.download('stopwords')
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')


def read_corpus(filename, use_lower=False, pos_tagging=False):
	with open(filename, "r", encoding="utf8") as f:
		tokens = word_tokenize(f.read())
		if use_lower:
			tokens = [token.lower() for token in tokens]
		if pos_tagging:
			return nltk.pos_tag(tokens)
		return tokens


def read_all_text_files(president_dir: str, use_lower=False, pos_tagging=False):
	"""
	:return: a list of lists: for each text document it contains the list of tokens of the text document
	"""
	data_path = "../data/{}/".format(president_dir)
	tokens_docs = []
	for filename in os.listdir(data_path):
		input_file = data_path + filename
		tokens_docs.append(read_corpus(input_file, use_lower, pos_tagging))
	return tokens_docs


def tfidf_per_doc(tokens_docs: List[List]):
	"""
	:param tokens_docs: list of tokens per document
	:return: List of dicts, where vocab is key and tfidf-score is value
	"""
	word_count_docs, dfs = count_words_per_doc(tokens_docs)

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


def count_words_per_doc(tokens_docs: List[List]):
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


def create_tf_dict(number_of_words: dict):
	"""
	Create a dictionary with the normalized tf scores per token, the tokens of the vocabulary are the keys, the tf scores are the values
	"""
	doc_size = sum([occ for _, occ in number_of_words.items()])
	return {token: occ / doc_size for token, occ in number_of_words.items()}


def top_n_per_document(tfidf_doc: dict, n=100):
	"""
	:return: returns a list of top n words sorted in descending order after the tfidf score of the document
	"""
	sorted_tuples = sorted(tfidf_doc.items(), key=lambda tup: tup[1], reverse=True)
	return [token for token, _ in sorted_tuples[:n]]


def read_speeches_as_text(file_path="bush"):
	"""
	:param file_path: path to speech files
	:return: List of speeches as strings
	"""
	data_path = "../data/{}/".format(file_path)
	documents = []
	for filename in os.listdir(data_path):
		with open(data_path+filename, "r") as f:
			documents.append(f.read())
	return documents


def euclidean_distance(x, y):
	return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

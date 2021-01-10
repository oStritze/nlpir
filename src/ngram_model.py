# https://github.com/lt616/NLP_trigram_model/blob/master/trigram_model.py

import utils
from collections import defaultdict
import random
from typing import List, Dict, Tuple


def create_ngram(president_dir: str, n=3):
	tokens_docs = utils.read_all_text_files(president_dir)
	return NgramModel(tokens_docs, n)


class NgramModel:
	def __init__(self, tokens_docs, n=3):
		self.N = n
		self.VOCAB = {"START", "END", "UNK"}
		self.total_word_count = 0
		self.ngram_counts = self.count_ngrams(tokens_docs)

	def count_ngrams(self, tokens_docs):
		"""
		Given a corpus iterator, populate dictionaries of ngram counts
		"""
		counts = [defaultdict(int) for _ in range(self.N)]
		for tokens in tokens_docs:
			for token in tokens:
				self.VOCAB.add(token)
			for i in range(self.N):
				for token in self.get_ngrams(tokens, i + 1):
					counts[i][token] += 1

			# Calculate total number of words
			self.total_word_count += len(tokens)

		# Set unigram start word count in higher ngram counts
		if self.N > 2:
			for i in range(1, self.N):
				counts[i][tuple(["START"] * (i + 1))] = counts[0][("START")]
		return counts

	def get_ngrams(self, sequence, n):
		"""
		Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
		This should work for arbitrary values of 1 <= size < len(sequence).
		"""
		results = []
		count_start = n - 1

		# Append "START" & "STOP" keywords to the sequence
		ext_sequence = []
		for i in range(0, count_start):
			ext_sequence += ["START"]
		ext_sequence += sequence + ["STOP"]

		for i in range(len(ext_sequence) - (n - 1)):
			result = []
			for j in range(i, i + n):
				result.append(ext_sequence[j])
			results.append(tuple(result))

		return results

	def smoothed_ngram_probability(self, ngram: Tuple[str]):
		"""
		Returns the smoothed ngram probability (using linear interpolation).
		"""
		lam = 1 / sum(range(self.N + 1))
		smoothed_prob = 0
		for i in reversed(range(self.N)):
			prev_gram = ngram[i:]
			smoothed_prob += (i + 1) * lam * self.raw_ngram_probability(prev_gram)
		return smoothed_prob

	def raw_ngram_probability(self, ngram: Tuple[str]):
		"""
		Returns the raw (unsmoothed) ngram probability
		"""
		i = len(ngram) - 1
		if i == 0:
			ngram = self.filter_UNK(ngram)
			return self.ngram_counts[i][ngram] / self.total_word_count
		else:
			prev_gram = self.filter_UNK(ngram[:i])
			ngram = self.filter_UNK(ngram)
			# Add 1 to handle UNK tokens
			return (self.ngram_counts[i][ngram] + 1) / (self.ngram_counts[i - 1][prev_gram] + 1)

	def filter_UNK(self, ngram):
		filtered = []
		for token in ngram:
			if not token in self.VOCAB:
				filtered.append("UNK")
			else:
				filtered.append(token)

		return tuple(filtered)

	def generate_speech(self, random_state: int, max_length=None):
		random.seed(random_state)

		prev_ngram = tuple(["START"] * self.N)
		speech = [prev_ngram[-1]]
		while speech[-1] != "END" and (max_length is None or len(speech) < max_length):
			ngrams = []
			probs = []
			for token in self.VOCAB:
				ngram = tuple(list(prev_ngram[1:]) + [token])
				ngrams.append(ngram)
				probs.append(self.smoothed_ngram_probability(ngram))
			next_ngram = random.choices(ngrams, weights=probs, k=1)[0]
			speech += [next_ngram[-1]]
			prev_ngram = next_ngram
		return " ".join(speech)

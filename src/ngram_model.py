# https://github.com/lt616/NLP_trigram_model/blob/master/trigram_model.py

import utils
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, Tuple


def create_ngram(president_dir: str, n=3, pos_n=3, use_lower=False, pos_tagging=False):
	tokens_docs = utils.read_all_text_files(president_dir, use_lower, pos_tagging)
	if pos_tagging:
		token_model = NgramModel([[token for token, pos_tag in tokens] for tokens in tokens_docs], n)
		pos_tag_model = NgramModel([[pos_tag for token, pos_tag in tokens] for tokens in tokens_docs], pos_n)
		tokens_per_pos = defaultdict(set)
		for tokens in tokens_docs:
			for token, pos_tag in tokens:
				tokens_per_pos[pos_tag].add(token)
		tokens_per_pos["START"].add("START")
		tokens_per_pos["STOP"].add("STOP")
		tokens_per_pos["UNK"].add("UNK")
		return token_model, pos_tag_model, tokens_per_pos

	return NgramModel(tokens_docs, n)


class NgramModel:
	def __init__(self, tokens_docs, n=3):
		self.N = n
		self.VOCAB = {"START", "STOP", "UNK"}
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
		lam = 1 / self.N
		smoothed_prob = 0
		for i in reversed(range(self.N)):
			prev_gram = ngram[i:]
			smoothed_prob += lam * self.raw_ngram_probability(prev_gram)
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


def generate_speech(random_state: int, token_model, pos_tag_model=None, tokens_per_pos=None, max_length=None, top_token=10, top_pos=10):
	random.seed(random_state)
	pos_tagging = bool(pos_tag_model is not None)

	prev_token_ngram = tuple(["START"] * token_model.N)
	if pos_tagging:
		prev_pos_ngram = tuple(["START"] * pos_tag_model.N)
	speech = [prev_token_ngram[-1]]

	while speech[-1] != "STOP" and (max_length is None or len(speech) < max_length):
		if pos_tagging:
			next_pos_ngram = predict_next_ngram(pos_tag_model, prev_pos_ngram, top_probs=top_pos)
			allowed_tokens = tokens_per_pos[next_pos_ngram[-1]]
			next_token_ngram = predict_next_ngram(token_model, prev_token_ngram, vocab=allowed_tokens, top_probs=top_token)

			prev_pos_ngram = next_pos_ngram
		else:
			next_token_ngram = predict_next_ngram(token_model, prev_token_ngram)

		speech += [next_token_ngram[-1]]
		prev_token_ngram = next_token_ngram
	return " ".join(speech)


def predict_next_ngram(model, prev_token_ngram, vocab=None, top_probs=10):
	ngrams = []
	probs = []
	if vocab is None:
		vocab = model.VOCAB
	for token in vocab:
		if token not in ["START", "UNK"]:
			ngram = tuple(list(prev_token_ngram[1:]) + [token])
			ngrams.append(ngram)
			probs.append(model.smoothed_ngram_probability(ngram))

	top_i = np.argsort(probs)[::-1][:top_probs]
	probs = np.array(probs)[top_i]
	probs /= np.max(probs)
	ngrams = np.array(ngrams)[top_i]
	return random.choices(ngrams, weights=probs, k=1)[0]

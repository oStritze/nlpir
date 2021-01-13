import utils
from scipy import spatial
import statistics
import re
from collections import Counter
from heapq import nsmallest
import math


def calc_cosine(orig_tfidf_docs, gen_tfidf_docs_adjusted, gen_speech_id_to_compare):
    """
    :param orig_tfidf_docs: tf_idf values for original speeches
    :param gen_tfidf_docs_adjusted: tf_idf values for generated speeches with added entries for missing values
    :param gen_speech_id_to_compare: ID of the generated speech to calculate the cosine similarity for
    :return: list of cosine similarities between the selected generated speech and all original speeches
    """
    cosine_similarities = []
    g = list(gen_tfidf_docs_adjusted[gen_speech_id_to_compare].values())

    for i in range(0, len(orig_tfidf_docs)):
        r = list(orig_tfidf_docs[i].values())
        cosine_similarities.append(1 - spatial.distance.cosine(r, g))

    return cosine_similarities


def get_cosine_sim_tfidf(orig_speeches_loc="bush", gen_speeches_loc="bush_generated", gen_speech_id_to_compare=None, print_results=True):
    """
        :param orig_speeches_loc: location of the txt files of the original speeches to analyse
        :param gen_speeches_loc: location of the txt files of the generated speeches to analyse
        :param gen_speech_id_to_compare: int if comparison to one generated speech, None if comparison over all
        :param print_results: Whether to print the calculated results in addition to returning them
        :return: mean cosine similarity, std of cosine similarities, list of cosine similarities
    """

    # calculate tf-idf for original speeches
    orig_tokens_docs = utils.read_all_text_files(orig_speeches_loc)
    orig_tfidf_docs = utils.tfidf_per_doc(orig_tokens_docs)

    # calculate tf-idf for generated speeches
    gen_tokens_docs = utils.read_all_text_files(gen_speeches_loc)
    gen_tfidf_docs = utils.tfidf_per_doc(gen_tokens_docs)

    # adjust for cosine similarity comparison by adding in missing words with value 0
    voc = set([val for sublist in orig_tokens_docs for val in sublist])
    gen_tfidf_docs_adjusted = [{key: gen_tfidf_docs[i].get(key, 0.0) for key in voc} for i in
                               range(0, len(gen_tfidf_docs))]

    # cosine similarity between 1 generated speech and all real speeches
    if gen_speech_id_to_compare is not None:
        cosine_similarities = calc_cosine(orig_tfidf_docs, gen_tfidf_docs_adjusted, gen_speech_id_to_compare)

        mean_cosine = sum(cosine_similarities) / len(cosine_similarities)
        std_cosine = statistics.stdev(cosine_similarities)

        if print_results:
            print("mean cosine similarity for generated speech " + str(gen_speech_id_to_compare) + ":", mean_cosine)
            print("standard deviation of cosine similarity for generated speech " + str(gen_speech_id_to_compare) + ":",
                  std_cosine)
        return mean_cosine, std_cosine, cosine_similarities

    # cosine similarity between all generated speeches and all real speeches
    else:
        all_cosine_similarities = []
        for i in range(0, len(gen_tfidf_docs_adjusted)):
            cosine_similarities = calc_cosine(orig_tfidf_docs, gen_tfidf_docs_adjusted, i)

            mean_cosine = sum(cosine_similarities) / len(cosine_similarities)
            all_cosine_similarities.append(mean_cosine)

        mean_all_cosine = sum(all_cosine_similarities) / len(all_cosine_similarities)
        std_all_cosine = statistics.stdev(all_cosine_similarities)

        if print_results:
            print("mean cosine similarity over all generated speeches:", mean_all_cosine)
            print("standard deviation of cosine similarity over all generated speeches:", std_all_cosine)
        return mean_all_cosine, std_all_cosine, all_cosine_similarities


def calculate_mean_sentence_length(speeches_loc="bush"):
    """
    :param speeches_loc: path to speech files
    :return: mean sentence length of all sentences of all speeches within speeches_loc
    """
    orig_speeches = utils.read_speeches_as_text(speeches_loc)
    sentence_lengths = []

    for speech in orig_speeches:
        sentences = re.split('\.|\!|\?', speech.replace("\n", ""))
        for sentence in sentences:
            if len(sentence) > 0:
                words = sentence.split(" ")
                sentence_lengths.append(len(words))

    mean_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    return mean_sentence_length


def calculate_mean_word_length(speeches_loc="bush"):
    """
    :param speeches_loc: path to speech files
    :return: mean word length of all words of all speeches within speeches_loc
    """
    orig_speeches = utils.read_speeches_as_text(speeches_loc)
    word_lengths = []

    for speech in orig_speeches:
        sentences = re.split('\.|\!|\?', speech.replace("\n", ""))
        for sentence in sentences:
            if len(sentence) > 0:
                words = sentence.split(" ")
                for word in words:
                    if len(word) > 0:
                        word_lengths.append(len(word))

    mean_sentence_length = sum(word_lengths) / len(word_lengths)
    return mean_sentence_length


def get_tfidf_ranklist(tfidf_doc, reverse=True):
    """
    :param tfidf_doc: tfidf dictionary to create a ranklist for
    :param reverse: list sort direction
    :return: ranklist of tfidf values
    """
    s_tfidf_doc = sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=reverse)

    rank, count, previous, result = 0, 0, None, {}
    for key, num in s_tfidf_doc:
        count += 1
        if num != previous:
            rank += count
            previous = num
            count = 0
        result[key] = rank

    return result


def get_combined_tfidf_ranklist(speeches_loc="bush"):
    """
    :param speeches_loc: path to speech files
    :return: combined tfidf ranklist over all speeches in speeches_loc
    """
    tfidf_docs = utils.tfidf_per_doc(utils.read_all_text_files(speeches_loc))
    ranklists = []
    for doc in tfidf_docs:
        ranklists.append(get_tfidf_ranklist(doc, reverse=True))

    summed_ranklist = sum(map(Counter, ranklists), Counter())
    combined_ranklist = get_tfidf_ranklist(summed_ranklist, reverse=False)
    return combined_ranklist


''' Alternative where we take the sum of tf-idf values over all documents instead of the sum of the ranklists
from collections import Counter

def get_combined_tfidf_ranklist(speeches_loc="bush"):
    tfidf_docs = utils.tfidf_per_doc(utils.read_all_text_files(speeches_loc))

    summed_tfidf_docs = sum(map(Counter, tfidf_docs), Counter())
    combined_ranklist = get_tfidf_ranklist(summed_tfidf_docs, reverse=True)
    return combined_ranklist
'''


def calc_top_n_distance(orig_ranklist, gen_ranklist, n=15):
    """
    :param orig_ranklist: ranklist for original speeches
    :param gen_ranklist: ranklist for generated speeches
    :param n: number of top words to use for calculation
    :return: euclidean distance between original and generated ranklists
    """
    top_n_words = nsmallest(n, orig_ranklist, key=orig_ranklist.get)

    gen_ranks = []
    for word in top_n_words:
        gen_ranks.append(gen_ranklist.get(word, max(gen_ranklist.values())))

    expected_ranks = list(range(1, n + 1))
    return utils.euclidean_distance(expected_ranks, gen_ranks)


def get_top_n_rank_distance(orig_speeches_loc="bush", gen_speeches_loc="bush_generated", gen_speech_id_to_compare=None, n=15):
    """
    :param orig_speeches_loc: path to original speech files
    :param gen_speeches_loc: path to generated speech files
    :param gen_speech_id_to_compare: int if comparison to one generated speech, None if comparison over all
    :param n: number of top words to use for calculation
    :return: euclidean distance between original and generated speeches
    """

    ranklist = get_combined_tfidf_ranklist(orig_speeches_loc)
    gen_tfidf_docs = utils.tfidf_per_doc(utils.read_all_text_files(gen_speeches_loc))

    if gen_speech_id_to_compare is None:
        distances = []
        for gen_doc in gen_tfidf_docs:
            ranklist_gen = get_tfidf_ranklist(gen_doc, reverse=True)
            distances.append(calc_top_n_distance(ranklist, ranklist_gen, n))
        return sum(distances) / len(distances)
    else:
        ranklist_gen = get_tfidf_ranklist(gen_tfidf_docs[gen_speech_id_to_compare], reverse=True)
        return calc_top_n_distance(ranklist, ranklist_gen, n)

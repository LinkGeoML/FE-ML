import os
import sys
import re

from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry

from external.datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler, monge_elkan, cosine, \
    strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies


sys.path.append(os.path.abspath('../Toponym-Matching'))


def getTMabsPath(ds):
    return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', ds)


def getRelativePathtoWorking(ds):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ds)


def get_langnm(s, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(s)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname


# Clean the string from stopwords, puctuations based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(s, stop_words=None, sorting=False, lang_detect=False):
    lname = get_langnm(s, lang_detect)

    tokens = wordpunct_tokenize(s)
    words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(lname)) if stop_words is None else set(stop_words)

    filtered_words = sorted_nicely(filter(lambda token: token not in stopwords_set, words)) if sorting else \
        filter(lambda token: token not in stopwords_set, words)
    stopped_words = sorted_nicely(filter(lambda token: token not in filtered_words, words)) if sorting else \
        filter(lambda token: token not in filtered_words, words)

    return filtered_words, stopped_words


def perform_stemming(s, lang_detect=False):
    try:
        lname = get_langnm(s, lang_detect)

        if lname in SnowballStemmer.languages:  # See which languages are supported
            stemmer = SnowballStemmer(lname)  # Choose a language
            s = stemmer.stem(s)  # Stem a word
    except KeyError:
        pass
        # print("Unicode error for {0}\n".format(e))

    return s


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# def enum(*sequential, **named):
#     enums = dict(zip(sequential, range(len(sequential))), **named)
#     reverse = dict((value, key) for key, value in enums.iteritems())
#     enums['reverse_mapping'] = reverse
#     return type('Enum', (), enums)


class StaticValues:
    algorithms = {
        'damerau_levenshtein': damerau_levenshtein,
        'davies': davies,
        'skipgram': skipgram,
        'permuted_winkler': permuted_winkler,
        'sorted_winkler': sorted_winkler,
        'soft_jaccard': soft_jaccard,
        'strike_a_match': strike_a_match,
        'cosine': cosine,
        'monge_elkan': monge_elkan,
        'jaro_winkler': jaro_winkler,
        'jaro': jaro,
        'jaccard': jaccard,
    }

    methods = [["Damerau-Levenshtein", 0.55],
               ["Jaro", 0.75],
               ["Jaro-Winkler", 0.7],
               ["Jaro-Winkler reversed", 0.75],
               ["Sorted Jaro-Winkler", 0.7],
               ["Permuted Jaro-Winkler", 0.7],
               ["Cosine N-grams", 0.4],
               ["Jaccard N-grams", 0.25],
               ["Dice bigrams", 0.5],
               ["Jaccard skipgrams", 0.45],
               ["Monge-Elkan", 0.7],
               ["Soft-Jaccard", 0.6],
               ["Davis and De Salles", 0.65]]

    nameIDs = {
        'damerau_levenshtein': 0,
        'davies': 12,
        'skipgram': 9,
        'permuted_winkler': 5,
        'sorted_winkler': 4,
        'soft_jaccard': 11,
        'strike_a_match': 8,
        'cosine': 6,
        'monge_elkan': 10,
        'jaro_winkler': 2,
        'jaro': 1,
        'jaccard': 7,
    }

    method_names = [
        "damerau_levenshtein",
        "jaccard",
        "jaro",
        "jaro_winkler",
        "jaro_winkler_reversed",
        "monge_elkan",
        "cosine",
        "strike_a_match",
        "soft_jaccard",
        "sorted_winkler",
        "skipgram",
        "davies"
    ]

import os
import sys
import re

from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry


sys.path.append(os.path.abspath('../Toponym-Matching'))

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

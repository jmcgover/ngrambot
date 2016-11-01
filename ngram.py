#! /usr/bin/env python3

# System
import sys
import os
from argparse import ArgumentParser
import json
from pprint import pformat

# Math
import random

# Data Structures
from collections import defaultdict

# Logging
import logging
from logging import handlers
LOGGER = logging.getLogger(__name__)
SH = logging.StreamHandler()
FH = logging.handlers.RotatingFileHandler(__name__ + ".log", maxBytes=5 * 1000000, backupCount = 5)
SH.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
FH.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(lineno)s:%(funcName)s:%(message)s"))
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(SH)
LOGGER.addHandler(FH)

# Natural Language Processing
from nltk import word_tokenize
from nltk import pos_tag
from nltk import everygrams
from nltk import ngrams

PUNCTUATION = set(",;.!?:'")
MONEY_SYMBOLS = set("$€£¥")
SENTENCE_END = set(".?!")

DESCRIPTION="""Generates ngrams based off of a corpus."""
def get_arg_parser():
    parser = ArgumentParser(prog=sys.argv[0], description=DESCRIPTION)
    parser.add_argument("-d", "--debug",
            action = "store_true",
            default = False,
        help="path (absolute or relative) to the data directory")
    parser.add_argument("-i", "--info",
            action = "store_true",
            default = False,
        help="path (absolute or relative) to the data directory")
    return parser

def save_dictionary_as_json(dictionary, filename, check = False):
    LOGGER.debug("Saving dictionary as JSON to '%s'", filename)
    if check and os.path.isfile(filename):
        LOGGER.warning("File already exists!")
        return False
    with open(filename, "w") as file:
        json.dump(dictionary, file)
    return True

def open_json_as_dictionary(filename, check = False):
    LOGGER.debug("Loading JSON as dictionary:'%s'", filename)
    if check and not os.path.isfile(filename):
        LOGGER.error("File doesn't exist!")
        return None
    with open(filename, "r") as file:
        return json.load(file)

def build_ngrams(tokens, low, high):
    LOGGER.debug("Building ngrams from %d to %d" % (low, high))
    assert low <= high
    assert low > 0
    grams = {}
    for n in range(low, high + 1):
        grams[n] = [g for g in ngrams(tokens, n)]
    return grams
def build_pos_ngrams(tagged, low, high):
    LOGGER.debug("Building POS ngrams from %d to %d" % (low, high))
    assert low <= high
    assert low > 0
    pos_tokens = []
    pos_words = defaultdict(list)
    for word, pos in tagged:
        pos_tokens.append(pos)
        pos_words[pos].append(word)
    grams = {}
    for n in range(low, high + 1):
        grams[n] = [g for g in ngrams(pos_tokens, n)]
    return grams, pos_words
def build_prefix_lookup(grams):
    LOGGER.debug("Building lookup table")
    lookup = {}
    for n, tuples in grams.items():
        LOGGER.debug("Building lookup table for %d" % n)
        lookup[n] = defaultdict(list)
        for t in tuples:
            for i in range(1, n):
                lookup[n][t[:i]].append(t[-i:])
                #LOGGER.debug("%d:%s:%s" % (i, t[:i], t[-1:]))
    #for n in lookup:
    #    LOGGER.debug("Lookup at %d" % n)
    #    for k,v in lookup[n].items():
    #        LOGGER.debug("%s:%s" % (k,v))
    return lookup
def find_starter_grams(grams):
    # Find Starter Words
    LOGGER.debug("Finding starter words")
    starter_words = set()
    starter_words.add(grams[2][0])
    for bigram in grams[2]:
        if bigram[0] in SENTENCE_END and bigram[1][0].isupper():
            starter_words.add(bigram[1])
    starter_grams = defaultdict(list)

    # Find Grams that Start with a Starter Word
    LOGGER.debug("Finding starter grams")
    for n, tuples in grams.items():
        for tuple in tuples:
            if tuple[0] in starter_words:
                starter_grams[n].append(tuple)
    return starter_grams
def combine_punctuation(tokens):
    LOGGER.debug("Combining punctuation:%s" % tokens)
    combined = []
    for ndx in range(1, len(tokens)):
        if tokens[ndx] in SENTENCE_END and ndx < len(tokens) - 1:
            tokens[ndx + 1] = tokens[ndx + 1].title()
        if tokens[ndx] in MONEY_SYMBOLS and ndX < len(tokens) - 1:
            combined.append(tokens[ndx] + tokens[ndx + 1])
        elif tokens[ndx] in PUNCTUATION:
            combined.append(tokens[ndx - 1] + tokens[ndx])
        elif tokens[ndx - 1] in PUNCTUATION:
            continue
        else:
            combined.append(tokens[ndx - 1])
    return combined
def generate_ngram_sentence(starter_grams, lookup, n):
    LOGGER.debug("Building N-Gram sentence using %d grams" % n)
    assert(n >= 2)
    assert(n <= len(lookup))
    word_list = []
    start_gram = random.choice(starter_grams[n])
    word_list.extend(start_gram)
    LOGGER.debug("Starting with: %s" % (start_gram,))
    while word_list[-1] not in SENTENCE_END:
        prefix = tuple(word_list[-(n - 1):])
        if prefix not in lookup[n]:
            LOGGER.error("Coudn't find %s in lookup" % (prefix,))
            LOGGER.debug("Trying to print lookup")
            for k,v in lookup[n].items():
                LOGGER.debug("%s:%s" % (k,v))
            break
        word_list.extend(random.choice(lookup[n][prefix])[-1:])
    return word_list
def generate_pos_ngram_sentence(starter_pos_ngrams, pos_lookup, pos_words, n):
    LOGGER.debug("Building POS sentence using %d grams" % n)
    pos_list = generate_ngram_sentence(starter_pos_ngrams, pos_lookup, n)

    if pos_list[0] in PUNCTUATION:
        LOGGER.warning("Removing %s" % pos_list.pop(0))
    LOGGER.debug("Tags:%s" % pos_list)
    capital_words = []
    for word in pos_words[pos_list[0]]:
        if word[0].isupper():
            capital_words.append(word)

    word_list = []
    if len(capital_words):
        word_list.append(random.choice(capital_words))
    else:
        word_list.append(random.choice(pos_words[pos_list[0]]).title())

    for tag in pos_list:
        word = random.choice(pos_words[tag])
        if "NN" not in tag and word != "I":
            word = word.lower()
        word_list.append(word)
    return word_list

class NGram(object):
    def make_ngram_sentence(self):
        assert False, "Please implement %s in %s" % (self.make_sentence, self.__class__.__name__)
        return
    def make_pos_sentence(self):
        assert False, "Please implement %s in %s" % (self.make_sentence, self.__class__.__name__)
        return

class Trump(NGram):
    def __init__(self, filename, low = 1, high = 6):
        self.filename = filename

        # Get Texts
        texts = []
        raw = ""
        for text in open_json_as_dictionary(filename)["texts"]:
            texts.append(text["text"])
            raw += "%s " % text["text"]

        LOGGER.debug("Tokenizing")
        tokens = word_tokenize(raw)
        tokens = [t for t in tokens if t != '”' and t != '"']
        LOGGER.debug("Tagging")
        pos = pos_tag(tokens)

        # N Gram Generation
        grams = build_ngrams(tokens, low, high)
        pos_grams, pos_words = build_pos_ngrams(pos, low, high)

        # N Gram Lookup Tables
        lookup = build_prefix_lookup(grams)
        pos_lookup = build_prefix_lookup(pos_grams)

        # Sentence Starting N Gram
        starter_grams = find_starter_grams(grams)
        starter_pos_grams = find_starter_grams(pos_grams)

        # Store Relevant Data
        self.texts          = texts
        self.raw            = raw
        self.grams          = grams
        self.pos_grams      = pos_grams
        self.pos_words      = pos_words
        self.lookup         = lookup
        self.pos_lookup     = pos_lookup
        self.starter_grams  = starter_grams
        self.starter_pos_grams  = starter_pos_grams
        return
    def make_ngram_sentence(self, n = 3):
        word_list = generate_ngram_sentence(self.starter_grams, self.lookup, n)
        combined = combine_punctuation(word_list)
        sentence = ' '.join(combined)
        LOGGER.debug("%d-Gram Sentence:%s" % (n,sentence))
        return sentence
    def make_pos_sentence(self, n = 6):
        word_list = generate_pos_ngram_sentence(self.starter_pos_grams, self.pos_lookup, self.pos_words, n)
        combined = combine_punctuation(word_list)
        sentence = ' '.join(combined)
        LOGGER.debug("%d-Gram POS Sentence:%s" % (n, sentence))
        return sentence

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Logging Information
    if args.info:
        SH.setLevel(logging.INFO)
    if args.debug:
        SH.setLevel(logging.DEBUG)

    # Make a Sentence
    meh = Trump("trumpTextRaw.json")
    ngram_sentence = meh.make_ngram_sentence()
    pos_sentence = meh.make_pos_sentence()
    print(ngram_sentence)
    print(pos_sentence)
    return 0

if __name__ == "__main__":
    LOGGER.info("Beginning Session")
    rtn = main()
    LOGGER.info("Ending Session")
    sys.exit(rtn)

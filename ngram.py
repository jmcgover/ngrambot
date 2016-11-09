#! /usr/bin/env python3

# System
import sys
import os
import pickle
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
FH = logging.handlers.RotatingFileHandler(os.path.splitext(sys.argv[0])[0] + ".log", maxBytes=5 * 1000000, backupCount = 5)
SH.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
FH.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(lineno)s:%(funcName)s:%(message)s"))
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(SH)
LOGGER.addHandler(FH)

GENSEN = logging.getLogger("generated")
GS = logging.FileHandler("generated" + ".log")
GS.setFormatter(logging.Formatter("%(asctime)s:%(message)s"))
GENSEN.setLevel(logging.INFO)
GENSEN.addHandler(GS)

# Natural Language Processing
from nltk import word_tokenize
from nltk import pos_tag
from nltk import everygrams
from nltk import ngrams

PUNCTUATION = set(",;.!?:'%")
PUNCTUATION.add("...")
MONEY_SYMBOLS = set("$€£¥")
SENTENCE_END = set(".?!")

DESCRIPTION="""Generates ngrams based off of a corpus."""
def get_arg_parser():
    parser = ArgumentParser(prog=sys.argv[0], description=DESCRIPTION)
    parser.add_argument("-i", "--info",
            help = "set console logging output to INFO")
    parser.add_argument("-d", "--debug",
            help = "set console logging output to DEBUG")
    parser.add_argument("-q", "--quiet",
            help = "set console logging output to ERROR (mostly quiet output)")
    parser.add_argument(
            metavar = "<text.json>",
            dest = "texts_filename",
            help = "JSON containing the texts to generate n-grams from")
    parser.add_argument(
            metavar = "<n>",
            type = int,
            dest = "n",
            help = "n value")
    parser.add_argument("-c", "--cache",
            action = "store_true",
            help = "caches NGram object based on JSON and loads if it already exists")
    parser.add_argument("-l", "--last",
            action = "store_true",
            help = "links on the last word of a gram only")
    parser.add_argument("-r", "--random",
            action = "store_true",
            help = "links on a random number ofthe last words")
    return parser

def save_as_json(object, filename, check = False):
    LOGGER.debug("Saving dictionary as JSON to '%s'", filename)
    if check and os.path.isfile(filename):
        LOGGER.warning("File already exists!")
        return False
    with open(filename, "w") as file:
        json.dump(object, file)
    return True

def open_json(filename, check = False):
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
                lookup[n][t[:i]].append(t)
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
    money_seen = False
    for ndx in range(1, len(tokens)):
        if tokens[ndx] in PUNCTUATION:
            combined.append(tokens[ndx - 1] + tokens[ndx])
        elif ndx < len(tokens) - 1 and tokens[ndx] in MONEY_SYMBOLS:
            tokens[ndx + 1] = tokens[ndx] + tokens[ndx + 1]
            money_seen = True
        elif tokens[ndx - 1] in PUNCTUATION:
            continue
        elif money_seen:
            money_seen = False
        else:
            combined.append(tokens[ndx - 1])
    return combined
def generate_ngram_sentence(starter_grams, lookup, n, link_type = None):
    LOGGER.debug("Building N-Gram sentence using %d grams" % n)
    assert(n >= 2)
    assert(n <= len(lookup))
    word_list = []
    start_gram = random.choice(starter_grams[n])
    word_list.extend(start_gram)
    LOGGER.debug("Starting with: %s" % (start_gram,))
    if link_type == "last":
        LOGGER.debug("Linking last token")
    elif link_type == "random":
        LOGGER.debug("Linking on random number of last tokens")
    else:
        LOGGER.debug("Linking last %d tokens" % (n - 1,))
    while word_list[-1] not in SENTENCE_END:
        prefix = None
        link_num = None
        if link_type == "last":
            prefix = tuple(word_list[-1:])
        elif link_type == "random":
            r = random.randrange(1, n)
            LOGGER.debug("Linking last %d tokens" % (r,))
            assert r != n, "The random number %d cannot equal n=%d" % (r, n)
            prefix = tuple(word_list[-r:])
        else:
            prefix = tuple(word_list[-(n - 1):])
        if prefix not in lookup[n]:
            LOGGER.error("Couldn't find %s in lookup for %d" % (prefix,n))
            continue
        choice = random.choice(lookup[n][prefix])
        extend = choice[len(prefix) - len(choice):]
        LOGGER.debug("Prefix:%s:Choice:%s:Extend:%s" % (prefix, choice, extend))
        word_list.extend(extend)
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

#class NGram(object):
#    def make_ngram_sentence(self):
#        assert False, "Please implement %s in %s" % (self.make_sentence, self.__class__.__name__)
#        return
#    def make_pos_sentence(self):
#        assert False, "Please implement %s in %s" % (self.make_sentence, self.__class__.__name__)
#        return

DELET = set((
    '"',
    "“",
    "”",
    "[",
    "]",
    "(",
    ")",
    ))
class NGram(object):
    def __init__(self, text, low = 1, high = 3, pos = False):

        LOGGER.debug("Tokenizing")
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in DELET]
        #LOGGER.debug("Tagging")
        #pos = pos_tag(tokens)

        # N Gram Generation
        grams = build_ngrams(tokens, low, high)

        # N Gram Lookup Tables
        lookup = build_prefix_lookup(grams)

        # Sentence Starting N Gram
        starter_grams = find_starter_grams(grams)

        # Store Relevant Data
        self.low            = low
        self.high           = high
        self.text           = text
        self.grams          = grams
        self.lookup         = lookup
        self.starter_grams  = starter_grams
        self.pos = pos

        if pos:
            pos_grams, pos_words = build_pos_ngrams(pos, low, high)
            pos_lookup = build_prefix_lookup(pos_grams)
            starter_pos_grams = find_starter_grams(pos_grams)
            self.pos_grams      = pos_grams
            self.pos_words      = pos_words
            self.pos_lookup     = pos_lookup
            self.starter_pos_grams  = starter_pos_grams
        return
    def make_ngram_sentence(self, n = 3, link_type = None):
        word_list = generate_ngram_sentence(self.starter_grams, self.lookup, n, link_type)
        combined = combine_punctuation(word_list)
        sentence = ' '.join(combined)
        LOGGER.debug("%d-Gram Sentence:%s" % (n,sentence))
        GENSEN.info("%d-Gram Sentence:%s" % (n,sentence))
        return sentence
    def make_pos_sentence(self, n = 6):
        assert self.pos, "POS NGrams not implemented"
        word_list = generate_pos_ngram_sentence(self.starter_pos_grams, self.pos_lookup, self.pos_words, n)
        combined = combine_punctuation(word_list)
        sentence = ' '.join(combined)
        LOGGER.debug("%d-Gram POS Sentence:%s" % (n, sentence))
        GENSEN.info("%d-Gram POS Sentence:%s" % (n, sentence))
        return sentence

def cache_filename(filename):
    return os.path.splitext(filename)[0] + "-ngram.pickle"


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Logging Information
    if args.info:
        SH.setLevel(logging.INFO)
    if args.debug:
        SH.setLevel(logging.DEBUG)
    if args.quiet:
        SH.setLevel(logging.ERROR)

    # Get Texts
    filename = args.texts_filename
    n = args.n
    generator = None
    if args.cache and os.path.isfile(cache_filename(filename)):
        LOGGER.debug("Loading from cache at %s" % cache_filename(filename))
        with open(cache_filename(filename), "rb") as file:
            generator = pickle.load(file)
    if generator == None or generator.high < n:
        LOGGER.debug("Getting Text") if generator == None else LOGGER.debug(
                "%d smaller than %d. Rebuilding." % (generator.high, n))
        texts = []
        raw = ""
        for text in open_json(filename):
            texts.append(text["text"])
        raw = '\n'.join(texts)
        for d in DELET:
            raw = raw.replace(d,"")
        generator = NGram(raw, high = n)
        with open(cache_filename(filename), "wb") as file:
            LOGGER.debug("Caching as %s" % cache_filename(filename))
            pickle.dump(generator, file)

    # Make a Sentence
    link_type = None
    if args.last:
        link_type = "last"
    if args.random:
        link_type = "random"
    ngram_sentence = generator.make_ngram_sentence(n, link_type)
    print(ngram_sentence)
    return 0

if __name__ == "__main__":
    LOGGER.info("Beginning Session")
    rtn = main()
    LOGGER.info("Ending Session")
    sys.exit(rtn)

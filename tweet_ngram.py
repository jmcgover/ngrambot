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

# NGram
import ngram

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

# Twitter
from twython import *

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
            metavar = "<oauth.json>",
            dest = "oauth_filename",
            help = "JSON containing the authorization necessary to twitter")
    parser.add_argument(
            metavar = "<text.json>",
            dest = "texts_filename",
            help = "JSON containing the texts to generate n-grams from")
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

OAUTH_KEYS = ("APP_KEY","APP_SECRET","OAUTH_TOKEN","OAUTH_TOKEN_SECRET")
def build_twitter(oauth):
    return Twython(
            oauth["APP_KEY"], oauth["APP_SECRET"],
            oauth["OAUTH_TOKEN"], oauth["OAUTH_TOKEN_SECRET"]
            )

def cache_filename(filename):
    return os.path.splitext(filename)[0] + "-ngram.pickle"

def trump(filename):
    # Get Texts
    n = random.choice((2,3,4))
    generator = None
    if os.path.isfile(cache_filename(filename)):
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
        for d in ngram.DELET:
            raw = raw.replace(d,"")
        generator = ngram.NGram(raw, high = n)
        with open(cache_filename(filename), "wb") as file:
            LOGGER.debug("Caching as %s" % cache_filename(filename))
            pickle.dump(generator, file)

    # Make a Sentence
    link_type = random.choice((None, "last", "random"))
    suffix = " @realDonaldTrump"
    tweet = None
    tries = 0
    while True:
        tries += 1
        ngram_sentence = generator.make_ngram_sentence(n, link_type)
        tweet = ngram_sentence + suffix
        if len(tweet) <= 140:
            LOGGER.debug("Try %d: Made a tweet of length %d after" % (tries, len(tweet)))
            break
        else:
            LOGGER.debug("Try %d: Tweet of %s chars is longer than %d" % (tries, len(tweet), 140))
    print(tweet)
    return tweet


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

    oauth = open_json(args.oauth_filename)
    if not oauth or any(key not in oauth for key in OAUTH_KEYS):
        LOGGER.critical("Failed to open %s. Existing file required." % args.oauth_filename)
    twitter = build_twitter(oauth)

    filename = args.texts_filename
    tweet = trump(filename)
    LOGGER.debug("Tweeting(%d chars):%s" % (len(tweet), tweet))
    twitter.update_status(status=tweet)
    return 0

if __name__ == "__main__":
    LOGGER.info("Beginning Session")
    try:
        rtn = main()
    except Exception as e:
        LOGGER.critical("Caught error: %s" % e)
        LOGGER.info("Failure. Exiting.")
        sys.exit(1)
    LOGGER.info("Ending Session")
    sys.exit(rtn)

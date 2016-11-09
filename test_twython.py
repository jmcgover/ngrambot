#! /usr/bin/env python3

# System
import sys
import os
from argparse import ArgumentParser
import json
from pprint import pformat

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
LOGGER.info("Beginning Session")
FH.setLevel(logging.DEBUG)

# Twitter
from twython import *

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

DESCRIPTION="""Tests the Twython API to see if it's garbage."""
def get_arg_parser():
    parser = ArgumentParser(prog=sys.argv[0], description=DESCRIPTION)
    parser.add_argument("-o", "--oauth",
            nargs = 1,
            default=["oauth.json"],
            help="path (absolute or relative) to the data directory")
    parser.add_argument("-d", "--debug",
            action = "store_true",
            default = False,
        help="path (absolute or relative) to the data directory")
    parser.add_argument("-i", "--info",
            action = "store_true",
            default = False,
        help="path (absolute or relative) to the data directory")
    return parser

def build_twitter(oauth):
    return Twython(
            oauth["APP_KEY"], oauth["APP_SECRET"],
            oauth["OAUTH_TOKEN"], oauth["OAUTH_TOKEN_SECRET"]
            )

def basic_test(twitter):
    verify_return = None
    LOGGER.info("Testing twitter authentication")
    verify_return = twitter.verify_credentials()
    LOGGER.debug("Verify Credentials Return: %s" % pformat(verify_return))
    LOGGER.info("Testing twitter authentication successful!")
    return verify_return

def get_timeline(twitter):
    timeline = None
    LOGGER.info("Gettining timeline")
    timeline = twitter.get_home_timeline()
    LOGGER.info("Gettining timeline successful!")
    return timeline

def main():
    parser  = get_arg_parser()
    args = parser.parse_args()
    SH.setLevel(logging.ERROR)
    if args.info:
        SH.setLevel(logging.INFO)
    if args.debug:
        SH.setLevel(logging.DEBUG)
    oauth_filename = args.oauth[0]
    oauth = open_json_as_dictionary(oauth_filename)
    LOGGER.info("Building Twitter")
    LOGGER.debug("OAuth Data: %s" % pformat(oauth))
    twitter = build_twitter(oauth)
    LOGGER.info("Building Twitter successful!")
    # basic_test(twitter)
    timeline = get_timeline(twitter)
    LOGGER.info("Number of Tweets:%d" % len(timeline))
    return 0

if __name__ == "__main__":
    rtn = main()
    sys.exit(rtn)

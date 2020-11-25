import logging
import sys


def config():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

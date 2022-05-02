#!/usr/bin/env python3
import re

# REFERENCES:
# [1] http://stackoverflow.com/questions/4998629/python-split-string-with-multiple-delimiters

def split(delimiters, string, maxsplit=0):
    """
    Tokenizes a string [1]
    """
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def remove_spaces(s):
    """
    Remove all spaces from a string
    :param s:
    :return:
    """
    return " ".join(s.split())
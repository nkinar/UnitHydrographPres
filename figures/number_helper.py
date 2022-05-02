#!/usr/bin/env python3

# Checks if something is a number

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



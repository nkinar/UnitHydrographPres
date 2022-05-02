#!/usr/bin/env python3
from number_helper import is_number
from check_list_tuple import check_list_tuple
from string_helper import split

# Matplotlib helper functions to construct a nice string to use in plotting

# Constants in this file
fs = 0.3            # default fixed space
beginning = '$'     # beginning for string
closing = '$'       # closing for string


def air_temp_label():
    return r'Air Temperature $T_a\hspace{0.3}(^{\circ}\mathrm{C})$'

def temp_label(small=False):
    if small:
        return r'$T\hspace{0.3}(^{\circ}\mathrm{C})$'
    return r'Temperature $\hspace{0.3}(^{\circ}\mathrm{C})$'

def degree_C_label():
    return r'$\hspace{0.3}(^{\circ}\mathrm{C})$'

def degree_C_label_without_brackets():
    return r'$^{\circ}\mathrm{C}$'

def simple_label(num, units):
    s = r'$' + str(num) + r'\hspace{0.3}\mathrm{' + units + '}$'
    return s

def _format_s(s):
    rv = r'\mathrm{' + str(s) + '}'
    return rv


def _hspace(num):
    rv = r'\hspace{' + str(num) + '}'
    return rv


def create_label(desc, terms, bracket=True):
    """
    Returns a formatted string with labels
    desc = description string 'Pressure'
    term = units 'kg m s^-1'  or ['kg', 'm', 's^-2' ]
    bracket = True to add brackets around the units
    """
    s = desc + '  '   # add space after the description
    N = len(terms)
    if N == 0:
        return s
    if check_list_tuple(terms) and N == 1:
        terms = terms[0]
    s += beginning
    if bracket:
        s += '('
    if isinstance(terms, str):
        terms = split('^ ', terms)
        N = len(terms)
    for k in range(N):
        if is_number(terms[k]):
            continue
        s += _hspace(fs) + _format_s(terms[k])
        if k < N-1 and is_number(terms[k+1]):
            s += '^{' + str(terms[k+1]) + '}'
    # append closing $ to string and bracket if necessary
    if bracket:
        s += _hspace(fs) + ')'
    s += closing
    # return the formatted string
    return s


def get_density_label():
    """
    Returns density label for use in a plot  (kg m^-3)
    """
    str =  '$\hspace{0.3}\mathrm{kg}\hspace{0.3}\mathrm{m}^{-3}$'
    return str


def test_string():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 15})
    # terms = ['kg','m','-1','s','-2']
    # terms = ['kg m s^-2']
    terms = 'kg m s^-2'
    # terms = 'kg m^-3'
    test = create_label('Pressure', terms)
    print(test)
    x = np.linspace(0, 10, 100)
    y = np.exp(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(test)
    plt.show()


def main():
    test_string()


if __name__ == '__main__': main()


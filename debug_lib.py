import re


_debug_levels = {
'Very Low':4,
'Low':3,
'Medium':2,
'High':1,
'Critical':0
}

GLOB_DEBUG_LEV = _debug_levels['Critical']

def get_match_from_docstring(function, regex_obj):
    """
    Title: Get the match from a function's docstring
    Input 1: Function
    Input 2: Regex object
    Output: The 1st string matching the regex
    """
    docstring = function.__doc__
    match_str = regex_obj.findall(docstring)[0]
    return match_str

def get_debug_lev(debug_lev):
    """
    Title: Get Debug Level
    Input: Debug Level Name
    Output: Debug Level
    """
    if debug_lev not in _debug_levels.keys():
        return _debug_levels['Critical']
    else:
        return _debug_levels[debug_lev]

def print_debug_msg(function_matrix, message, debug_level):
    """
    Title: Print Debug Messages
    Input 1: Function Matrix - Functions executed to get the primary message
    Input 2: Secondary Message
    Input 3: Debug Level
    Output: None
    """
    if debug_level <= GLOB_DEBUG_LEV:
        debug_msg = ''
        for function, args in function_matrix.items():
            debug_msg += function(*args)
        message = message.replace('\n','\n\t')
        print '{0}\n\t{1}'.format(debug_msg, message)

def print_debug_msg_lev_crit(function_matrix, message):
    """
    Title: Print Critical Debug Messages
    Input 1: Function Matrix
    Input 2: Secondary Message
    Output: None
    """
    print_debug_msg(function_matrix, message, get_debug_lev('Critical'))

def print_debug_msg_lev_high(function_matrix, message):
    """
    Title: Print Debug Level High Debug Messages
    Input 1: Function Matrix
    Input 2: Secondary Message
    Output: None
    """
    print_debug_msg(function_matrix, message, get_debug_lev('High'))


def print_debug_msg_lev_medium(function_matrix, message):
    """
    Title: Print Debug Level Medium Debug Messages
    Input 1: Function Matrix
    Input 2: Secondary Message
    Output: None
    """
    print_debug_msg(function_matrix, message, get_debug_lev('Medium'))


def print_debug_msg_lev_low(function_matrix, message):
    """
    Title: Print Debug Level Low Debug Messages
    Input 1: Function Matrix
    Input 2: Secondary Message
    Output: None
    """
    print_debug_msg(function_matrix, message, get_debug_lev('Low'))


def print_debug_msg_lev_very_low(function_matrix, message):
    """
    Title: Print Debug Level Very Low Debug Messages
    Input 1: Function Matrix
    Input 2: Secondary Message
    Output: None
    """
    print_debug_msg(function_matrix, message, get_debug_lev('Very Low'))

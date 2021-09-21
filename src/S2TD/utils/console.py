import numpy as np
from datetime import datetime

SEP_LARGE = "-" * 62
SEP_SMALL = "-" * 62


def print_start(message, verbose):
    if verbose:
        msg = time_message(message)
        print("%s\n%s\n%s" % (SEP_LARGE, msg, SEP_LARGE))


def print_end(message, elapsed, verbose):
    """
    :param message: str
    :param elapsed: datetime.timedelta
    :param verbose: bool
    :return:
    """
    if verbose:
        try:
            seconds = elapsed.seconds
            minutes = np.round(seconds / 60, 2)
            elapsed_msg = "%s minute" % minutes if minutes == 1 else "%s minutes" % minutes
        except AttributeError:
            elapsed_msg = "unknown"
        msg = time_message(message)
        print("%s\n%s\n%s\n%s" % (SEP_LARGE, msg, time_message("Elapsed: " + elapsed_msg), SEP_LARGE))


def print_status(message, verbose):
    if verbose:
        print(time_message(message))


def print_status_sep(message, verbose):
    if verbose:
        print(SEP_SMALL + "\n" + time_message(message))


def time_message(message):
    return "|%s| %s" % (what_time_is_it(), message)


def what_time_is_it():
    return datetime.now().isoformat()

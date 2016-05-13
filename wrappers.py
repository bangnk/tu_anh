from functools import wraps
import sys
import datetime as dt


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()


def log_and_call(location):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            save_std = sys.stdout
            logger = Logger(location % (args[0], args[1]))
            sys.stdout = logger
            sys.stderr = logger
            result = func(*args, **kwargs)
            logger.close()
            sys.stdout = save_std
            return result
        return wrapper
    return decorator


def elapsed():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = dt.datetime.now()
            result = func(*args, **kwargs)
            elapsed_time = dt.datetime.now() - start_time
            print '%s(): %is' % (func.__name__, elapsed_time.seconds)
            return result
        return wrapper
    return decorator
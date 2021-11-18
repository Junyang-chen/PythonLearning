from time import time, sleep
from functools import wraps

def timeit_no_argument(func):
    # bla = timeit_no_argument(bla)
    # result = bla(*args)
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("before timeit no args")
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print("after timeit no args")
        print(f"{func.__name__} took {end - start:.4f} seconds to run.")
        return result
    return wrapper

def timeit(logger=None):
    # bla = timeit_no_argument(decorator_args)(bla)
    # result = bla(*args)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("before timeit")
            start = time()
            result = func(*args, **kwargs)
            end = time()
            print("after timeit")
            print_msg = f"{func.__name__} took {end - start:.4f} seconds to run."
            if logger:
                logger.info(print_msg)
            else:
                print(print_msg)
            return result
        return wrapper
    return decorator

@timeit()
@timeit_no_argument
def testing():
    sleep(1)

if __name__ == '__main__':
    testing()
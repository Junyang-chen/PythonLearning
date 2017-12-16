"""
python Cookbook learning code
"""

# 1.4 binary heap in python
import heapq
import random
nums = [2*i+1 for i in range(7)]
random.shuffle(nums)
# get top 3 max number
maxheap = heapq.nlargest(3, nums)
print("Max heap with {0} is:".format(3))
print(maxheap)
# heapify
heapq.heapify(nums)

# 1.6 default dictionary
from collections import defaultdict
d = defaultdict(list)
d['ID'].append(1)
d['ID'].append(2)
d['rank'].append(2)
d['rank'].append(1)

# 1.7 ordered dictionary
from collections import OrderedDict
d = OrderedDict()
d['1'] = 1
d['2'] = 2
d['3'] = 3

# 1.12 Counter
from collections import Counter
a = [1,1,2,3,1,5]
d = Counter(a)

# 1.13
rows = [
        {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
        {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
        {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
        {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}
]
from operator import itemgetter
rows_by_fname = sorted(rows, key=itemgetter('fname'))
rows_by_fname_lambda = sorted(rows, key=lambda x:x['fname'])

# 1.16 filter
def filter_func(item):
    if item >0 and item % 2 == 0:
        return True
    else:
        return False
a = [1,2,3,4,5]
a = filter(filter_func, a)
print(a)

# 1.17 subset of dict
prices = {
       'ACME': 45.23,
       'AAPL': 612.78,
       'IBM': 205.55,
       'HPQ': 37.20,
       'FB': 10.75
}
p1 = {key:value for key, value in prices.items() if value > 200}
print(p1)

# 2.1 regular expression
line = 'asdf fjdk; afed, fjek,asdf, foo'
import re
print(re.split(r'[;,\s]\s*', line))

# 2.2 string startswith & endswith
s = r'www.helper.com'
print(s.startswith(r'www'))
print(s.startswith((r'www', r'html',r'ftp')))

# 2.3 match wild card
from fnmatch import fnmatch, fnmatchcase
print(fnmatch('test.txt', '*.txt'))

# 2.5 searching and replace text
date = 'Today is 11/29/2012, we are very excited'
import re
print(re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2', date))

# 2.10 space strip
s = ' hello world \n'
print(s.strip())
print(s.lstrip())
print(s.rstrip())

t = '-----hello====='
print(t.lstrip('-'))
print(t.rstrip('='))

# string interpolate
s = '{name} has {n} messages.'
s.format(name='Guido', n=37)

# text wrap to fixed number of column
import textwrap
s = "Look into my eyes, look into my eyes, the eyes, the eyes, \ the eyes, not around the eyes, don't look around the eyes, \ look into my eyes, you're under."
print(textwrap.fill(s, 70))
print(textwrap.fill(s, 30))

# 3.2 decimal calculation
from decimal import Decimal
a = Decimal('4.2')
b = Decimal('3.6')
a + b
print(a + b)

# 3.7 inf and nan
a = float('inf')
b = float('nan')
print(a, b)

# 4.2 iterable

class iterable:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return iter(self.value)

a = iterable([1,2,3])
for i in a:
    print(i)

# 4.9 iterate through combination or permutation
from itertools import permutations, combinations
items = ['a', 'b', 'c']
for i in permutations(items):
    print(i)

for i in permutations(items, 2):
    print(i)

for i in permutations(items,1):
    print(i)

for i in combinations(items, 2):
    print(i)

# 4.10 iterate through index value pair
for i, v in enumerate(['a', 'b', 'c']):
    print("{0}th element is {1}".format(i+1, v))


for i, v in enumerate(['a', 'b', 'c'], 1):
    print("{0}th element is {1}".format(i, v))

# 4.11 zip function of iterating over two iterables
a = [1, 2, 3]
b = ['a', 'b', 'c', 'd']
for x, y in zip(a,b):
    print(x, y)

# 4.12 iterate on seperate items
from itertools import chain
a = set([1,2,3])
b= set([4,5,6])
for i in chain(a,b):
    print(i)

# 4.14 nested container
import collections
items = [1, 2, [3, 4, [5, 6], 7], 8]
items2 = ['Dave', 'Paula', ['Thomas', 'Lewis']]

def flatten(container, ignore_type=(str)):
    for i in container:
        if isinstance(i, collections.Iterable) and not isinstance(i, ignore_type):
            # this has to be yield from as flatten(i) is a generator
            yield from flatten(i)
        else:
            yield i

print(list(flatten(items)))
print(list(flatten(items2)))

# 4.13 generator to read file pipline
import os
import fnmatch
import re
# find files with format
def gen_file(file_format, dir):
    for filename in os.listdir(dir):
        if fnmatch.fnmatch(filename, file_format):
            yield os.path.join(dir, filename)

def gen_open(filenames):
    for file in filenames:
        with open(file, 'rt') as f:
            yield f

def gen_concatenate(iterators):
    for i in iterators:
        yield from i

def gen_grep(pattern, lines):
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line

logfiles = gen_file('log*.txt', os.path.join(os.getcwd(), 'Samplefiles'))
file_opener = gen_open(logfiles)
lines = gen_concatenate(file_opener)
matched_lines = gen_grep(r'go{2}d', lines)
for line in matched_lines:
    print(line)

# 5.12
import os
os.path.exists(r'Regex.py')
os.path.isfile(r'Regex.py')
os.path.isdir(r'Regex.py')
os.listdir(os.getcwd())

# 6.1 open csv
import csv
with open(r'Samplefiles/sample.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    print(headers)
    for row in f_csv:
        print(row)

# 6.2 JSON
import json

data = {
'name' : 'ACME',
'shares' : 100,
'price' : 542.23
}
# dumps and loads are for objects in cache
json_str = json.dumps(data)
print(json_str)
data = json.loads(json_str)

# with easiler read
print(json.dumps(data, indent=4))

# dump and load are for .json file
with open(r'Samplefiles/1.json', 'w') as f:
    json.dump(data, f)

with open(r'Samplefiles/1.json', 'r') as f:
    data = json.load(f)

# turns a json into a python object
class JSONobject:
    def __init__(self, d):
        self.__dict__ = d

s = json.dumps(data)
data = json.loads(s, object_hook=JSONobject)
data.name

# 6.8 RDBMS
import sqlite3
db = sqlite3.connect('database.db')
c = db.cursor()
c.execute('create table portfolio (symbol text, shares integer, price real)')
db.commit()

stocks = [
('GOOG', 100, 490.1),
('AAPL', 50, 545.75),
('FB', 150, 7.45),
('HPQ', 75, 33.2),
]
c.executemany('insert into portfolio values (?,?,?)', stocks)
db.commit()

for row in db.execute('select * from portfolio where price >= ?', (1,)):
    print(row)

# 7.2 function with only keyword, a bare * is forced to have the behind para be keyword arguments only
def recv(*, block):
    pass

recv(1,2,3, block=True)

# 7.3 adding additional information for function
def add(x:int, y:int) -> int:
    return x + y

# no type checking will happen, only shown as help()
print(add(3,1.5))
help(add)

# 7.7 lambda
# lambd constant is evaluated at run time
x = 10
a = lambda y: x+y
x = 20
b = lambda  y:x+y
print(a(10))
print(b(10))
x = 3
print(a(10))

# to fix x as compile time
a = lambda y,x=x: x+y
print(a(10))
x = 3
print(a(10))

funcs = [lambda x: x + n for n in range(4)]
for fun in funcs:
    print(fun(0))

funcs = [lambda x, n=n: x + n for n in range(4)]
for fun in funcs:
    print(fun(0))

# 7.11 inline callback function

def apply_async(func, args, *, callback):
    # Compute the result
    result = func(*args)

    # Invoke the callback with the result
    callback(result)

# Inlined callback implementation
from queue import Queue
from functools import wraps

class Async:
    def __init__(self, func, args):
        self.func = func
        self.args = args

def inlined_async(func):
    @wraps(func)
    def wrapper(*args):
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            result = result_queue.get()
            try:
                a = f.send(result)
                apply_async(a.func, a.args, callback=result_queue.put)
            except StopIteration:
                break
    return wrapper

# Sample use
def add(x, y):
    return x + y

@inlined_async
def test():
    r = yield Async(add, (2, 3))
    print(r)
    r = yield Async(add, ('hello', 'world'))
    print(r)
    for n in range(10):
        r = yield Async(add, (n, n))
        print(r)
    print('Goodbye')

if __name__ == '__main__':
    # Simple test
    print('# --- Simple test')
    test()

    print('# --- Multiprocessing test')
    import multiprocessing
    pool = multiprocessing.Pool()
    apply_async = pool.apply_async
    test()


# 8.3 making object support context manager

class RAII(object):
    def __init__(self, pt):
        self.pt = pt
        print("Initializing pointer!")

    def __enter__(self):
        return self.pt

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.pt
        print("Pointer is destoryed!")

raii = RAII([])
with raii as r:
    r.append(1)
    r.append(2)
    print(r)
print(raii.pt)

# 8.4 saving memory of a large number of instance
class Date:
    __slots__ = ['year', 'month', 'day']
    def __init__(self):
        pass

# 8.5 Property with getter and setter
import math
class circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius**2

    @property
    def permeter(self):
        return 2*math.pi * self.radius

# 8.11
class Structure:
    _fields = []
    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError("Length should be the same!")
        for name, arg in zip(self._fields, args):
            setattr(self, name, arg)

class Stock(Structure):
    _fields = ['name', 'price', 'share']

s = Stock('ACME', 50, 91.1)

# 8.12 abstract class
from abc import ABCMeta, abstractmethod
class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass
    @abstractmethod
    def write(self, data):
        pass

class SocketStream(IStream):
    def read(self, maxbytes=-1):
        pass
    def write(self, data):
        pass

# 8.14 make some iterable
class class_iterable:
    def __init__(self, content):
        self.content = content

    def __iter__(self):
        return iter(self.content)

a = class_iterable([1,2,3])
for i in a:
    print(i)

# 8.15 defining more than one constructor
import datetime

class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __repr__(self):
        return "{0}-{1}-{2}".format(self.year, self.month, self.day)

    @classmethod
    def today(cls):
        today = datetime.datetime.now().date()
        return cls(today.year, today.month, today.day)

a = Date(2017, 11, 30)
b = Date.today()
print(a, b)
d = Date.__new__(Date)
print(dir(d))

# 8.16 extend class with mixins
class LoggingMappingMixin:
    __slots__ = ()
    def __getitem__(self, key):
        print('Getting key:'.format(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        print('Setting item:{0} to {1}'.format(key, value))
        super().__setitem__(key, value)

class LoggingDict(LoggingMappingMixin, dict):
    pass

a = LoggingDict()
a[1] = 2


# 8.23 using reference that won't increase reference counting
# weakref module
import weakref

class Node:
    def __init__(self, value):
        self._value = value
        self._parent = None
        self.children = []

    def __repr__(self):
        return "Node:{}".format(self._value)

    @property
    def parent(self):
        return self._parent if self._parent is None else self._parent()

    @parent.setter
    def parent(self, node):
        # self._parent = node
        self._parent = weakref.ref(node)

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

root = Node('parent')
a = Node('child')
root.add_child(a)
print(a.parent)
del root
print(a.parent)
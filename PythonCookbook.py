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



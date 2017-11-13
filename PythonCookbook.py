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
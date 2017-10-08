"""
Served as learning and practising of regular expressions
"""
from collections import Counter
import re
"""
reading "Gone with the wind"
"""

list_word = []
with open('Gone_with_the_Wind-Margaret_Mitchell.txt', 'r') as file:
    list_word = file.read().split()
text = ' '.join(list_word)
print("Length of novel is {0}".format(len(list_word)))
counter = Counter(list_word)
print("Most common words are:")
print(counter.most_common(10))

# look for I * you
pattern1 = r'[I]\s\w+\s\w+'
pattern2 = r'\bTHE\s\bEND'
p = re.compile(pattern1)
result = p.findall(text)
print(result)

p = re.compile('[I]\s\w+\s\w+')
m = p.match('I love you')
m.group()


import re

# ?= is the positive lookahead assertion
# it won't consume the three group
# for example, the baab is found, but b will not be returned, therefore second one will match on baab
s = r'abaabaabaabaae'
format_match = r'([^aeiouAEIOU+\-])([aeiouAEIOU]{2,})(?=[^aeiouAEIOU+\-])'
comp = re.compile(format_match)
itr = comp.finditer(s)
found = 0
for i in itr:
    found = 1
    print(i.group(2))
if found == 0:
    print(-1)

"""
Hackerrank app 1: Detect html
https://www.hackerrank.com/challenges/detect-html-links/problem
"""

import re

n = 2
str_list = [r'<p><a href="http://www.quackit.com/html/tutorial/html_links.cfm">Example Link</a></p>',
            r'<div class="more-info"><a href="http://www.quackit.com/html/examples/html_links_examples.cfm">More Link Examples...</a></div>']
format_word = r'<a href=.*<\/a>'
format_link = r'(href=)(")(.*?)(")'
format_not_name = r'(<[^<>]*>)'
com_word = re.compile(format_word)
comp_link = re.compile(format_link)
comp_name = re.compile(format_not_name)
for i in range(n):
    html = str_list[i]
    word = com_word.search(html)
    if word:
        link = comp_link.search(html)
        name = comp_name.findall(html)
        if link and name:
            link = link.group(3)
            for i in name:
                html = html.replace(i,'')
            print("{0},{1}".format(link.strip(),html.strip()))
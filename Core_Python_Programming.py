"""
__repr__ vs __str__

__repr__ is used for debugging to be unambigous, for developers
__str__ is for user

__init__, __new__   New is called before init()
"""

class Complex(object):
    count = 0
    def __new__(cls, *args, **kwargs):
        obj = super(Complex, cls).__new__(cls)
        cls.count += 1
        return obj

    def __init__(self, real, imag):
        self.real=real
        self.imag=imag

    def __repr__(self):
        return "Complex class real {0}, imag {1}".format(self.real, self.imag)

    def __str__(self):
        return "{0} + {1}i".format(self.real, self.imag)



"""
Descriptor
13.16.4

__get__(), __set__(), __delete()

"""

class descriptor(object):
    def __init__(self, name=""):
        self.__name = name

    def __get__(self, instance, type=None):
        print("Oops! can't get this varible.")

    def __set__(self, instance, value):
        print("Can't set {0}".format(value))

class Contain(object):
    foo = descriptor()

d = Contain()

d.foo

"""
property getter setter
"""

class complex(object):

    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value * 2

a = complex(2)
a.x = 3
print(a.x)

"""
Raw_input vs Input()
in 3.5 input is behave as raw_input,
to evaluate input use eval(input())
"""

a = input("Please enter a list")
b = eval(input("Please enter a list"))
print(type(a))
print(type(b))

"""
enable function checking in base class
__init_subclass__ is used in python 3.6
https://docs.python.org/3/reference/datamodel.html
"""

class BaseMeta(type):
    def __new__(cls, name, bases, body):
        if name != 'Base' and 'bar' not in body:
            raise TypeError('bad user class!')
        return super().__new__(cls, name, bases, body)

class Base(metaclass=BaseMeta):

    def foo(self):
        return self.bar()

class Derived(Base):
    def bar(self):
        pass

b = Derived()
"""
regular expression
"""

import re
m = re.match('re', 're')
if m is not None: m.group()


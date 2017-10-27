"""
For learning pandas example
"""


class Base1(object):
    def __init__(self, size):
        print("In Base1 __init__")
        self.size = size

    def sizeof(self):
        print("In Base1")
        return self.size

class Base2(object):
    def __init__(self, size):
        print("In Base2 __init__")
        self.size=size

    def sizeof(self):
        print("in Base2")
        return self.size

class Derive(Base1, Base2):
    def __init__(self, size1, size2):
        self.size1, self.size2 = size1, size2
        super(Derive, self).__init__(size1)

    def sizeof(self):
        return super(Derive,self).sizeof()


"""
only first one will be called
"""


a=Derive(1,2)
a.sizeof()
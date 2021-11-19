
class Mymeta(type):
    def __init__(self, name, bases, dic):
        super().__init__(name, bases, dic)
        print("==>MyMeta.__init")
        print(self.__name__)
        print(dic)
        print(self.yaml_tag)

    def __new__(cls, *args, **kwargs):
        print("==>Mymata.__new__")
        print(cls.__name__)
        return type.__new__(cls, *args, **kwargs)

    def __call__(cls, *args, **kwargs):
        print("==>Mymeta.__call__")
        obj = cls.__new__(cls)
        cls.__init__(cls, *args, **kwargs)
        return obj

class Foo(metaclass=Mymeta):
    yaml_tag = "Foo!"

    def __init__(self, name):
        print("Foo.__init__")
        self.name = name

    def __new__(cls, *args, **kwargs):
        print("Foo.__new__")
        return super(Foo, cls).__new__(cls, *args, **kwargs)

foo = Foo('foo')
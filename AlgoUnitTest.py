import unittest
import Algo

def assertListEqual(list1, list2):
    if len(list1) != len(list2): return False
    for i in range(len(list2)):
        if list1[i]!=list2[i]:
            return False
    return True

sortFuncs = [getattr(Algo, func) for func in dir(Algo) if func.endswith('Sort')]
class SortAlgoTest(unittest.TestCase):
    def test_empty(self):
        for func in sortFuncs:
            self.assertListEqual(func([]), [])

    def test_singleElementList(self):
        for func in sortFuncs:
            self.assertListEqual(func([1]), [1])

    def test_sortedList(self):
        for func in sortFuncs:
            self.assertListEqual(func([1,2,3,4]), [1,2,3,4])

    def test_normalList1(self):
        for func in sortFuncs:
            self.assertListEqual(func([5,2,1,4,6]), [1,2,4,5,6])

    def test_normalList2(self):
        for func in sortFuncs:
            self.assertListEqual(func([8,4,5,6,9,1]), [1,4,5,6,8,9])

class SecondMiniTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(Algo.FindSecondMini([]), None)

    def test_singleElementList(self):
        self.assertEqual(Algo.FindSecondMini([1]), 1)

    def test_normalList1(self):
        self.assertEqual(Algo.FindSecondMini([1,3,6,-1]), 1)

class MaxSubarrayTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(Algo.MaxSubArray([]), None)

    def test_singleElementList(self):
        self.assertEqual(Algo.MaxSubArray([1]), 1)

    def test_normalList1(self):
        self.assertEqual(Algo.MaxSubArray([1,3,6,-1]), 10)
        self.assertEqual(Algo.MaxSubArray([1,3,5,-7,-9,2,4,-6,8]), 9)

if __name__ == '__main__':
    unittest.main()

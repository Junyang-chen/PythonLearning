""""
This script is serverd as a practise of Algorithm and data structure learning in Python
"""

# def FindSecondMini(input_list):
#     if not isinstance(input_list, list):
#         raise ValueError("input must be list type!")
#     if len(input_list)==0:
#         return None
#     if len(input_list)==1:
#         return input_list[0]
#     minimum = float('inf')
#     sec_minimum = minimum
#     for i in input_list:
#         if i<minimum:
#             sec_minimum = minimum
#             minimum = i
#         elif i<sec_minimum:
#             sec_minimum = i
#     return sec_minimum
#
# def MaxSubArray(input_list):
#     if not isinstance(input_list, list):
#         raise ValueError("input must be list type!")
#     if len(input_list)==0:
#         return None
#     if len(input_list)==1:
#         return input_list[0]
#     maximum = 0
#     max_scan = 0
#     for i in input_list:
#         max_scan = max_scan + i
#         if max_scan < 0:
#             max_scan = 0
#         if max_scan > maximum:
#             maximum = max_scan
#     return maximum
#
#
# def InsertionSort(input_list):
#     if not isinstance(input_list, list):
#         raise ValueError("Input must be list type!")
#
#     if len(input_list) <=1:
#         return input_list
#
#     for i in range(1,len(input_list)):
#         j = i-1
#         temp = input_list[i]
#         while(temp < input_list[j] and j>=0):
#             input_list[j+1] = input_list[j]
#             j -= 1
#         input_list[j+1] = temp
#     return input_list
#
#
# def MergeSort(input_list):
#     def mergeHelper(li, p, q):
#         if p<q:
#             r = (p+q)//2
#             mergeHelper(li, p, r)
#             mergeHelper(li, r+1, q)
#             Merge(li, p, q, r)
#
#     def Merge(li, p, q, r):
#         n1 = r-p+1
#         n2 = q-r
#         left_list = [0]*n1
#         right_list = [0]*n2
#         for i in range(n1):
#             left_list[i] = li[p+i-1]
#         for j in range(n2):
#             right_list[j] = li[r+j]
#         i,j = 0,0
#         k=p
#         while i<n1 and j<n2:
#             if left_list[i]<right_list[j]:
#                 li[k-1] =left_list[i]
#                 i+=1
#             else:
#                 li[k - 1] = right_list[j]
#                 j += 1
#             k += 1
#         if i==n1:
#             li[k-1:q]=right_list[j:]
#         else:
#             li[k-1:q]=left_list[i:]
#
#
#     mergeHelper(input_list, 1, len(input_list))
#     return input_list
#
# def HeapSort(input_list):
#     return MaxHeap(input_list).heapSort()
#
# class MaxHeap(object):
#     def __init__(self, values):
#         self.list = list(values)
#         self.length = len(values)
#
#     def _getParentIndex(self, index):
#         return (index + 1) // 2 -1
#
#     def _getLeftChildIndex(self, index):
#         return (index + 1) * 2 - 1
#
#     def _getRightChildIndex(self, index):
#         return (index + 1)* 2
#
#     def heapify(self, index):
#         largest = index
#         l = self._getLeftChildIndex(index)
#         r = self._getRightChildIndex(index)
#         if l < self.heapSize and self.list[l] > self.list[largest]:
#             largest = l
#         if r < self.heapSize and self.list[r] > self.list[largest]:
#             largest = r
#         if largest != index:
#             temp = self.list[largest]
#             self.list[largest] = self.list[index]
#             self.list[index] = temp
#             self.heapify(largest)
#
#     def _buildInitialHeap(self):
#         self.heapSize = self.length
#         for i in reversed(range(self.length//2)):
#             self.heapify(i)
#
#     def heapSort(self):
#         self._buildInitialHeap()
#         for i in reversed(range(self.length)):
#             temp = self.list[0]
#             self.list[0] = self.list[i]
#             self.list[i] = temp
#             self.heapSize -= 1
#             self.heapify(0)
#         return list(self.list)
#
#
# def QuickSort(input_list):
#     if not isinstance(input_list, list):
#         raise ValueError("Input must be list type!")
#
#     if len(input_list) <=1:
#         return input_list
#
#     def partition(input_list, p, r):
#         x = input_list[r]
#         i = p-1
#         for j in range(p, r):
#             if input_list[j] <= x:
#                 i += 1
#                 temp = input_list[j]
#                 input_list[j] = input_list[i]
#                 input_list[i] = temp
#         input_list[r] = input_list[i+1]
#         input_list[i+1] = x
#         return i + 1
#
#     def QuickSortHelper(input_list, p, r):
#         if p<r:
#             q = partition(input_list, p, r)
#             QuickSortHelper(input_list, p, q-1)
#             QuickSortHelper(input_list, q, r)
#
#     QuickSortHelper(input_list, 0, len(input_list)-1)
#
#     return input_list
#
# def FindMedian(input_list):
#     def partition(input_list, p, r):
#         pivot = input_list[p-1]
#         i = p
#         for j in range(p, r):
#             if input_list[j] <= pivot:
#                 input_list[j], input_list[i] = input_list[i], input_list[j]
#                 i += 1
#         input_list[i-1] , input_list[p-1] = input_list[p-1] , input_list[i-1]
#         return i
#
#     if not input_list:
#         return None
#     elif len(input_list) == 1:
#         return input_list[0]
#
#     def FindMedianHelper(input_list, p, r, k):
#         if p==r:
#             return input_list[p-1]
#         q = partition(input_list, p, r)
#         q_relative = q-p+1
#         if q_relative == k:
#             return input_list[q-1]
#         elif q_relative>k:
#             return FindMedianHelper(input_list, p, q-1, k)
#         elif q_relative<k:
#             return FindMedianHelper(input_list, q+1, r, k-q_relative)
#
#
#     ind = len(input_list) // 2
#     return FindMedianHelper(input_list, 1, len(input_list), ind+1)
#
#
#
# def log2nRecursion(n):
#     return 1 + log2nRecursion(n//2) if n>1 else 0
#
# def log2Iteration(n):
#     log = 0
#     while n > 1:
#         n //= 2
#         log += 1
#     return log
#
# print(log2nRecursion(3), log2Iteration(3))
# print(log2nRecursion(7), log2Iteration(7))
# print(log2nRecursion(8), log2Iteration(8))
# print(log2nRecursion(64), log2Iteration(64))
# print(log2nRecursion(65), log2Iteration(65))
#
#
# def printInorder(root):
#     if root:
#         printInorder(root.left)
#         print(root.val)
#         printInorder(root.right)
#
#
# def tribonacci(n):
#     if (n == 0):
#         return 0
#     elif (n == 1 or n == 2):
#         return 1
#     return tribonacci(n-2) + tribonacci(n-1) + tribonacci(n-3)
#
#
# mem_list = [0] * 38
# def tribonacciWithMem(n):
#     if (n == 0):
#         mem_list[0] = 0
#         return 0
#     elif (n == 1 or n == 2):
#         mem_list[n] = 1
#         return 1
#     else:
#         if mem_list[n] != 0:
#             return mem_list[n]
#         else:
#             mem_list[n] = tribonacciWithMem(n-2) + tribonacciWithMem(n-1) + tribonacciWithMem(n-3)
#         return mem_list[n]
#
# print(tribonacciWithMem(25))
#
# def tribonacciIteration(n):
#     if (n == 0):
#         return 0
#     elif (n == 1 or n == 2):
#         return 1
#     else:
#         T3 = 0
#         T2 = T1 = 1
#         for i in range(3,n+1):
#             tribonacci = T1 + T2 + T3
#             T3 = T2
#             T2 = T1
#             T1 = tribonacci
#
#         return tribonacci
# print(tribonacciIteration(25))
#
#
# def minmaxGasDist(stations, K):
#     left, right = 0, stations[-1] - stations[0]
#     while (left + 10e-6 < right):
#         mid = (left + right) / 2
#         count = 0
#
#         for i in range(len(stations) - 1):
#             count += (stations[i + 1] - stations[i]) // mid
#
#         if count > K:
#             left = mid
#         else:
#             right = mid
#
#     return right
#
#
# st = [5, 8, 10, 25, 28, 31, 72, 80, 85, 100]
# K = 8
# print(minmaxGasDist(st, K))
#
#
# class solution:
#     def gas_dis(self, lis, m):
#         dist = [lis[i] - lis[i - 1] for i in range(1, len(lis))]
#         if lis[0] > 0:
#             dist.append(lis[0])
#         if lis[-1] < m:
#             dist.append(m - lis[-1])
#         dist.sort()
#         return dist
#
#     def minmaxGasDist(self, dist, t):
#             lo = 0
#             hi = st[-1]
#             while lo < hi:
#                 mi = lo + (hi - lo) // 2
#                # print(lo, mi, hi)
#                 cnt = 0
#                 for i in range(len(dist)):
#                     if dist[i] > mi:
#                         cnt += dist[i] // mi + (dist[i] % mi > 0) - 1
#                 if cnt <= t:
#                     hi = mi
#                 else:
#                     lo = mi + 1
#             return lo
#
# # test:
# st= [5, 8, 10, 25, 28, 31, 72, 80, 85, 100]
# m = 125
# t = 8
# solution = solution()
# dist = solution.gas_dis(st, m)
# print("dist:", dist)
# print("minmaxGasDist:", solution.minmaxGasDist(dist, t))
#
# class Node:
#     def __init__(self, val):
#         self.val = val
#         self.prev = None
#         self.next = None
#     def printNode(self):
#         root = self
#         while (root):
#             print(root.val)
#             root  = root.next
#
# def recursiveSort(root: Node) -> Node:
#     if not root or (not root.prev and not root.next):
#         return root
#     next = root.next
#     root.next = root.prev
#     root.prev = next
#     if not root.prev:
#         return root
#     return recursiveSort(root.prev)
# root1 = Node(1)
# root2 = Node(2)
# root3 = Node(3)
# root4 = Node(4)
# root5 = Node(5)
# root1.next = root2
# root2.prev = root1
# root2.next = root3
# root3.prev = root2
# root3.next = root4
# root4.prev = root3
# root4.next = root5
# root5.prev = root4
# a = recursiveSort(root1)
# a.printNode()


class Solution:
    def readBinaryWatch(self, num: int):
        avaliableLight = [i for i in range(11)]
        result = []
        eachCombination = []
        self.getNFromAll(avaliableLight, eachCombination, 1, result)
        return self.convertCombinationToTime([len(i) == num for i in result])

    def getNFromAll(self, input, eachCombination, next: int, result):
        result.append(eachCombination.copy())
        for i in range(next, len(input) + 1):
            eachCombination.append(input[i - 1])
            self.getNFromAll(input, eachCombination, next + 1, result)
            eachCombination.pop()
        return

    def convertCombinationToTime(result):
        hours = [1, 2, 4, 8]
        mins = [1, 2, 4, 8, 16, 32]
        hour = 0
        minute = 0
        str_result = []
        for combination in result:
            for num in combination:
                if (num < 5):
                    hour += hours[num]
                else:
                    minute += mins[num]
            if (hour > 11 or minute > 59):
                continue
            minute = str(minute) if minute > 9 else '0' + str(minute)
            str_result.append(str(hour)+ ':' + minute)
        return str_result

solution = Solution()
solution.readBinaryWatch(4)


# implement a queue from a stack

class queue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, val):
        self.stack1.append(val)

    def dequeue(self):
        if not self.stack1 and not self.stack2:
            print('Queue is empty')
            return
        if (not self.stack2):
            while (self.stack1):
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


q = queue()
for i in range(5):
    q.enqueue(i)

for i in range(5):
    print(q.dequeue())

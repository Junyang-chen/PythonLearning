""""
This script is serverd as a practise of Algorithm and data structure learning in Python
"""

def FindSecondMini(input_list):
    if not isinstance(input_list, list):
        raise ValueError("input must be list type!")
    if len(input_list)==0:
        return None
    if len(input_list)==1:
        return input_list[0]
    minimum = float('inf')
    sec_minimum = minimum
    for i in input_list:
        if i<minimum:
            sec_minimum = minimum
            minimum = i
        elif i<sec_minimum:
            sec_minimum = i
    return sec_minimum

def MaxSubArray(input_list):
    if not isinstance(input_list, list):
        raise ValueError("input must be list type!")
    if len(input_list)==0:
        return None
    if len(input_list)==1:
        return input_list[0]
    maximum = 0
    max_scan = 0
    for i in input_list:
        max_scan = max_scan + i
        if max_scan < 0:
            max_scan = 0
        if max_scan > maximum:
            maximum = max_scan
    return maximum


def InsertionSort(input_list):
    if not isinstance(input_list, list):
        raise ValueError("Input must be list type!")

    if len(input_list) <=1:
        return input_list

    for i in range(1,len(input_list)):
        j = i-1
        temp = input_list[i]
        while(temp < input_list[j] and j>=0):
            input_list[j+1] = input_list[j]
            j -= 1
        input_list[j+1] = temp
    return input_list


def MergeSort(input_list):
    def mergeHelper(li, p, q):
        if p<q:
            r = (p+q)//2
            mergeHelper(li, p, r)
            mergeHelper(li, r+1, q)
            Merge(li, p, q, r)

    def Merge(li, p, q, r):
        n1 = r-p+1
        n2 = q-r
        left_list = [0]*n1
        right_list = [0]*n2
        for i in range(n1):
            left_list[i] = li[p+i-1]
        for j in range(n2):
            right_list[j] = li[r+j]
        i,j = 0,0
        k=p
        while i<n1 and j<n2:
            if left_list[i]<right_list[j]:
                li[k-1] =left_list[i]
                i+=1
            else:
                li[k - 1] = right_list[j]
                j += 1
            k += 1
        if i==n1:
            li[k-1:q]=right_list[j:]
        else:
            li[k-1:q]=left_list[i:]


    mergeHelper(input_list, 1, len(input_list))
    return input_list
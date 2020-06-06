
input1 = [
    [1,3],
    [2,4]
]
output1 = 2.5

input2 = [
    [1,3],
    [2]
]
output2 =2

input3 = [
    [1,3,5,7,9],
    [2,4,6,8,10,12,14,16,18,20]
]
output3 = 8

def medianOfTwoSortedArray(array1, array2):
    n = len(array1)
    m = len(array2)
    k = (n + m )//2 +1
    i = j = 0
    m1 = m2 =-1
    for p in range(k):
        m2 = m1
        if(i < n and j < m):
            if (array1[i]<array2[j]):
                m1 = array1[i]
                i += 1
            else:
                m1 = array2[j]
                j += 1
        elif (i==n):
            m1 = array2[j]
            j += 1
        else:
            m1 = array1[i]
            i += 1
    return m1 if (n + m )%2 ==1 else (m1 + m2)/2



assert(medianOfTwoSortedArray(*input1) == output1)
assert(medianOfTwoSortedArray(*input2)== output2)
assert(medianOfTwoSortedArray(*input3)== output3)

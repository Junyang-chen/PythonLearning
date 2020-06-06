

input1 = [3, [[1,3],[2,3]]]
output1 = 2

input2 = [3, [[1,2],[2,3],[3,1]]]
output2 = -1

def minimumSemesters(N, relations):
    from collections import defaultdict
    from queue import Queue
    graph = defaultdict(set)
    inCount = [0]*(N+1)
    for relation in relations:
        graph[relation[0]].add(relation[1])
        inCount[relation[1]] += 1
    q = Queue()
    for i in range(1,N+1):
        if inCount[i] == 0:
            q.put(i)
    course = semester = 0
    while q.qsize():
        l = q.qsize()
        course += l
        for i in range(l):
            cur = q.get()
            for avaiableCourse in graph[cur]:
                inCount[avaiableCourse] -= 1
                if (inCount[avaiableCourse] == 0):
                    q.put(avaiableCourse)
        semester += 1
    return semester if course == N else -1

print(minimumSemesters(*input1))
print(minimumSemesters(*input2))
import bisect
def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
    count, s = 0, 0
    sorted_sums = [0]
    for x in nums:
        s += x
        l = bisect.bisect_left(sorted_sums, s - upper)
        r = bisect.bisect_right(sorted_sums, s - lower)
        count += r - l
        bisect.insort(sorted_sums, s)
    return count

countRangeSum([-2,5,-1], -2, 2)
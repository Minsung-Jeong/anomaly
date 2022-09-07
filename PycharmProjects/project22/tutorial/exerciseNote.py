# 10000 보다 작은 수, 666포함한 모든 경우의 수 만들고 -> 줄세운 뒤 선택

# N = int(input())

array = [8, 4, 6, 2, 9, 1, 3, 7, 5]
N = len(array)
# bubble 정렬
for i in range(N-1):
    for j in range(N-i-1):
        if array[j] > array[j+1]:
            array[j], array[j+1] = array[j+1], array[j]

# selection 정렬 한 바퀴 돌면서 가장 작은 값 가장 앞으로
array = [8, 4, 6, 2, 9, 1, 3, 7, 5]
N = len(array)
for i in range(N):
    min_idx = i
    for j in range(i+1, N):
        if array[j] < array[min_idx]:
            min_idx = j
    array[i], array[min_idx] = array[min_idx], array[i]

# merge 정렬

array = [8,4,6,2,9,1,3,7,5]

def merge_sort(array):
    if len(array) < 2:
        return array

    mid = len(array) // 2
    low_arr = merge_sort(array[:mid])
    high_arr = merge_sort(array[mid:])

    merged_arr = []
    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    print(merged_arr)
    return merged_arr

print("before: ",array)
array = merge_sort(array)
print("after:", array)


# 병합정렬
array = [8,4,6,2,9,1,3,7,5]
def merge_sort(array):
    print('------시작-----', array)
    if len(array) < 2:
        print('--return-- array < 2', array)
        return array

    mid = len(array) // 2
    print('중간 길이:', mid)

    print('start low', array[:mid])
    low_arr = merge_sort(array[:mid])

    print('start high', array[mid:])
    high_arr = merge_sort(array[mid:])

    merged_arr = []
    l = h = 0
    print('while시작 전 low, high', low_arr, high_arr)
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    print('merged while문-00', merged_arr)
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    print('merged array',merged_arr)
    return merged_arr

a = merge_sort(array)



def merge_sort(array):
    if len(array) < 2:
        return array

    mid = len(array) // 2

    r_arr = merge_sort(array[:mid])
    l_arr = merge_sort(array[mid:])

    merged_arr = []
    r = l = 0
    while r < len(r_arr) and l < len(l_arr):
        if r_arr[r] < l_arr[l]:
            merged_arr.append(r_arr[r])
            r += 1
        else:
            merged_arr.append(l_arr[l])
            l += 1
    merged_arr += l_arr[l:]
    merged_arr += r_arr[r:]
    return merged_arr

array = [8,4]
merge_sort(array)


# bfs
graph_list = {1: set([3, 4]),
              3: set([1, 5, 8]),
              4: set([1, 9]),
              5: set([3]),
              6: set([7]),
              7 : set([9,6]),
              8: set([3]),
              9: set([4, 7])}


def dfs(graph, root):
    visited = []
    stack = [root]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            # 노드의 하위노드 - visited
            stack += graph[n] - set(visited)
    return visited








# ---------------------
n=5
graph=[[] for _ in range(n+1)]
graph[1].append(4)
graph[1].append(2)
graph[2].append(3)
graph[2].append(4)
graph[3].append(4)

edge = [[1,4],[1,2],[2,3],[2,4],[3,4]]
graph = {}
for i in range(len(edge)):
    node1, node2 = edge[i]
    if node1 not in graph:
        graph[node1] = node2
    elif node2 not in graph[node1]:
        graph[node2] = node1

#----------------------------------------

class tree:
    def __init__(self, value=None):
        if value is not None:
            self.value = value
        else:
            self.value = None
        self.right = None
        self.left = None


Ntree = tree(10)
Ntree.left = tree(5)
Ntree.left.right = tree(2)

Ntree.right = tree(5)
Ntree.right.right = tree(1)
Ntree.right.right.left = tree(-1)


stack = [(Ntree, Ntree.value)]
result = float('inf')
curr = Ntree.value
while stack:
    node, curr = stack.pop()


    if node.left is None and node.right is None:
        if result > curr:
            result = curr

    if node.left:
        stack.append((node.left, node.left.value+curr))
    if node.right:
        stack.append((node.right, node.right.value+curr))

import numpy as np
data = np.array([3,1,2])


class SparseArray:
    def __init__(self, arr, size):
        self.size = size
        self.map = {}

        orig_arr_size = len(arr)
        for i, e in enumerate(arr):
            if i >= orig_arr_size:
                break
            if e != 0:
                map[i] = e




arr  =[ 0,5,0]
size =5
map = {}

orig_arr_size = len(arr)
for i, e in enumerate(arr):
    if i >= orig_arr_size:
        break
    if e != 0:
        map[i] = e
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

graph_list = {1: set([3, 4]),
              3: set([1, 5, 8]),
              4: set([1, 9]),
              5: set([3]),
              6: set([7]),
              7 : set([9,6]),
              8: set([3]),
              9: set([4, 7])}

# 뒤에 것 부터 뽑으면 dfs, 앞에 것 부터 뽑으면 bfs
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
dfs(graph_list, 1)


from collections import deque
def BFS(graph, root):
    visited = []
    que = deque([root])

    while que:
        n = que.popleft()
        if n not in visited:
            visited.append(n)
            que += graph[n] - set(visited)
    return visited
BFS(graph_list, 1)


#최단 경로 문제
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


# sparse array 문제, # 0아닌 값의 [행번호, 열번호, 값]으로 변환하는
arr =[0,0,5]
size =8

class SparseArray:
    def __init__(self, arr, size):
        self.size = size
        self.map = {}
        orig_arr_size = len(arr)
        for i, e in enumerate(arr):
            if i >= orig_arr_size:
                break
            if e != 0:
                self.map[i] = e

    def set(self, i, v):
        self.check_bounds(i)
        self.map[i] = v

    def get(self, i):
        self.check_bounds(i)
        v = self.map.get(i)
        if v is None:
            return 0
        return v

    def check_bounds(self, i):
        if i < 0 or i >= self.size:
            raise IndexError()

# 문제 77, 정렬되지 않은 리스트
given = [(5, 8), (1, 3), (20, 25), (4, 10)]
# 시작 숫자 기준으로 오름차순 정렬
for i in range(len(given)):
    for j in range(len(given)-i-1):
        if given[j][0] > given[j+1][0]:
            given[j], given[j+1] = given[j+1], given[j]

result = given.copy()
# 인터벌 합치기 1. 겹치는 부분 체크하기(겹치기, 포함), 포함 관계가 3개?
for i in range(len(given)-1):
    j = i+1
    i_start, i_end = given[i]
    j_start, j_end = given[j]

    # i가 j를 포함
    if i_start <= j_start and i_end >= j_end:
        result.remove(j)
    # i가 앞에서 j와 겹침
    if i_start <= j_start and i_end <= j_end:
        del result[i]
        del result[i]
        result.insert(i, (i_start, j_end))
    # j가 i를 포함
    if i_start >= j_start and i_end <= j_end:
        del result[i]
    # j가 앞에서 i와 겹침
    if i_start >= j_start and i_end >= j_end:
        del result[i]
        del result[i]
        result.insert(i, (j_start, j_end))


# problem77 해답
given = [(5, 12), (1, 3), (20, 25), (4, 10)]
given = sorted(given)
merged_intervals = []

for interval in given:
    if merged_intervals and interval[0] < merged_intervals[-1][1]:
        # merged_intervals.append는 잘못된 것 - 추가가 아니라 업데이트 형식이어야 지워야하는 것 자동으로 지울 수 있다
        merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))
    else:
        merged_intervals.append(interval)



# problem133
class tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def get_path(self, target):
        path = [self]
        while True:
            node = path[-1]

            if target < node.value:
                if not node.left:
                    return None
                path.append(node.left)
            elif target > node.value:
                if not node.right:
                    return None
                path.append(node.right)
            else:
                return path
    def get_most_left(self, node):
        while node and node.left:
            print("node.left:", node.left.value)
            node = node.left
        return node

    def get_next_bigger(self, value):
        path = self.get_path(value)
        node = path[-1]
        print("node right value", node.right.value)
        most_left_of_right = self.get_most_left(node.right)
        if most_left_of_right:
            print("most_left logic")
            return most_left_of_right
        # path 바로 이전 = parent, parent.left가 있으면 그게
        for parent in path[:-1][::-1]:
            if parent.left:
                print("parent left logic")
                return parent


# new_tree = tree(10)
# new_tree.left = tree(5)
# new_tree.right = tree(30)
# new_tree.right.left = tree(22)
# new_tree.right.right = tree(35)
#
# new_tree.get_next_bigger(30).value

new_tree = tree(10)
new_tree.left = tree(5)
new_tree.right = tree(30)
new_tree.right.left = tree(22)
new_tree.right.left.right = tree(23)
new_tree.right.right = tree(35)

new_tree.get_next_bigger(22).value



s = "[](){}"
s = "}]()[{"
# s = "{{{"

score = 0
for i in range(len(s)):
    new_s = s[i:]+s[:i]

    st = ''
    nd = ''
    rd = ''
    for x in new_s:
        if x =='(' or x==')':
            st += x
            if len(st) == 2 and st  == '()':
                st =''
        if x =='[' or x==']':
            nd += x
            if len(nd) == 2 and nd  == '[]':
                nd =''
        if x =='{' or x=='}':
            rd += x
            if len(rd) == 2 and rd == '{}':
                rd =''
    if len(st)+len(nd)+len(rd) == 0:
        score += 1

print(score)

x = bin(7)[2:]
x
for i in
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


# 예제----------------------------------

def fib(N):
    if N <= 1:
        return N
    result = fib(N-1)+fib(N-2)
    return result


# n,m : n에 m 이 존재하는지 확인

N = 5
X = [4, 1, 5, 2 ,3]
M = 5
Y = [1, 3, 7, 9, 5]


target = Y[0]

X.sort()

left = 0
right = len(X) - 1

while left <= right:
    mid = (left+right)//2
    print(left, right)
    if X[mid] == target:
        print('mid', mid)
        break
    elif X[mid] > target:
        right = mid
    else:
        left = mid + 1

# 0-----------

n = 5
arr = [4, 1, 5, 2 ,3]
m= 5
item = [1, 3, 7, 9, 5]


arr.sort()
def bin_search(arr, target):
    left = 0
    right = len(arr)-1
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:

            return mid
        elif arr[mid] > target:
            right = mid
        else:
            left = mid + 1

for target in item:
    if bin_search(arr, target) !=  None:
        print(1)
    else:
        print(0)
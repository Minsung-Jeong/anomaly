
from collections import Counter

li = [1, 1, 2, 3, 1, 2, 1, 5,5]
f = Counter(li)
ans = [-1 for i in range(len(li))]
idx = 0


while li:
    milestone = li.pop(0)
    for i in range(len(li)):
        if f[milestone] < f[li[i]]:
            ans[idx] = li[i]
            break
    idx += 1
print(*ans)


from collections import Counter

n = 7
nums = [1, 1, 2, 3, 4, 2, 1]
nums_count = Counter(nums)

result = [-1 for i in range(n)]
stack = [0]











# ----------------------------------
from collections import Counter
from sys import stdin

n = 9
nums = [1, 1, 2, 3, 1, 2, 1, 5,5]
nums_count = Counter(nums)
result = [-1] * n
stack = [0]

for i in range(1, n):
    print(i,'번째')
    while stack and nums_count[nums[stack[-1]]] < nums_count[nums[i]]:
            print(stack)
            result[stack.pop()] = nums[i]
    stack.append(i)

print(*result)
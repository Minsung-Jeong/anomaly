
n = 3
# input = '26 40 83 49 60 57 13 89 99'
input = '1 100 100 100 1 100 100 100 1'
arr = [int(x)for x in input.split()]
a = []
for i in range(1,n+1):
    a.append(arr[(i-1)*3 : i*3])

for i in range(1, n):
    a[i][0] = a[i][0] + min(a[i-1][1], a[i-1][2])
    a[i][1] = a[i][1] + min(a[i-1][0], a[i-1][2])
    a[i][2] = a[i][2] + min(a[i-1][1], a[i-1][0])


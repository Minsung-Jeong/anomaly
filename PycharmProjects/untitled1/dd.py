total_sp = 121
skills = [[1,2],[1,3],[3,6],[3,4],[3,5]]


count = 0

score = [0 for i in range(len(skills)+1)]
child = [0 for i in range(len(skills)+1)]

for skill in skills:
    child[skill[0]-1] +=1

for i in range(len(skills)+1):
    if child[i] == 0:
        score[i] = 1

count = 0

while 0 not in score:
    for skill in skills:
        if score[skill[1]-1] == count:
            score[skill[0]-1] += count


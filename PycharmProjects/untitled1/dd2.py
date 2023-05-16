def bfs(graph, start_node):
    visit = []
    queue = []
    queue.append(start_node)
    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.extend(graph[node])
    return visit

######입력
total_sp = 13*10
skills = [[10,8],[10,4],[10,11],[4,3],[4,9],[8,7],[3,6],[9,5]]

### 데이터 정리
skill_set = set()
child_set = set()
for skill in skills:
    skill_set.update(skill)
    child_set.add(skill[1])

# 차집합
root = list(skill_set - child_set)[0]
# set 을 list로 변형
skill_set = list(skill_set)

skill_dict = {sk:[] for sk in skill_set}

for i in skill_set:
    for sk_ord in skills:
        if sk_ord[0] == i:
            skill_dict[i].append(sk_ord[1])

skill_score = {sk:0 for sk  in skill_set}


## root 찾기

hierarchy = bfs(skill_dict, root)
hierarchy.reverse()


#leaf부터
# leaf = []


for item in hierarchy: # leaf 찾고 그에 대해 1pt 부여
    print(item)
    if len(skill_dict[item]) == 0:
        skill_score[item] = 1
        # leaf.append(item)
    else:
        pass

for sk_prs in hierarchy:
    for child in skill_dict[sk_prs]:
        skill_score[sk_prs] += skill_score[child]


# score 랑 순서 등을 매핑하는 부분
skill_set.sort()
# 그냥 순서대로
score_result = [skill_score[x] for x in skill_set]

# int / float 에 대한 조건 잘 기억
x_value = int(total_sp/sum(score_result))

import numpy as  np
total_result = (np.array(score_result)* x_value).tolist()

skills = [[10,8],[10,4],[10,11],[4,3],[4,9],[8,7],[3,6],[9,5]]

skill_li = []
for i in range(len(skills)):
    for j in range(len(skills[i])):
        skill_li.append(skills[i][j])
skill_set = list(set(skill_li))
skill_dict = {sk:[] for sk  in skill_set}
for i in skill_set:
    for sk_ord in skills:
        if sk_ord[0] == i:
            skill_dict[i].append(sk_ord[1])




# graph = {부모 : [자식1, ..., 자식n]}
def dfs(graph, start_node):
    visit = [] # 방문 노드 리스트
    stack = []

    stack.append(start_node) # 시작 노드를 stack에 삽입

    while stack:
        node = stack.pop()
        if node not in visit: # 이미 방문한 경우에는 탐색하지 않을 것
            visit.append(node) # 방문한 노드를 기록
            stack.extend(graph[node]) # graph[node] = 해당 node의 자식노드



    return visit
a = dfs(skill_dict, 10)


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



[5, 6, 9, 3, 7, 11, 4, 8, 10]
item = 10
if len(skill_dict[item]) == 0:
    skill_score[item] = 1
    leaf.append(item)
    hierarchy.remove(item)
else:
    pass

print(skill_score)
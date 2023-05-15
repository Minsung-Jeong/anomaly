from itertools import permutations
part_times = [[3,6,3],[2,4,2],[10,12,8],[11,15,5],[1,8,10],[12,13,1]]
# part_times = [[1,2,1],[1,2,2],[2,3,1],[3,4,1],[1,4,2]]



# len(a)
# per = list(product(part_times))
# per
# index_m = 0
# exit_m = 0
# combine = []
#
# money = 0
# money_t = 0
#
#
# for i in range(len(part_times)):
#     if part_times[index_m][0] >= exit_m: # 시작일이 이전 종료일 보다 늦을 때만 작동하도록 설정
#
#         money_t += part_times[index_m][2]
#         exit_t = part_times[index_m][1]
#         index_t = i+1
#
#         if index_t < len(part_times):
#             print('a')
#         else:
#             combine.append(money_t)


income = []
permuted_pt = list(permutations(part_times, len(part_times)))
while len(part_times) != 0:
    try:
        component = list(permuted_pt.pop())
    except:
        break

    money_ = 0
    index_ = 0
    exit_ = 0
    for cell in component:
        if cell[0] >= exit_:
            money_ += cell[2]
            exit_ = cell[1]

    income.append(money_)


max(income)
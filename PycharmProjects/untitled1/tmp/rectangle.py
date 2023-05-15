# # rectangle 1
#
# v = [[1,4],[3,4],[3,10]]
#
#
# x_ = []
# y_ = []
#
#
#
#
# for coor in v:
#     if coor[0] in x_:
#         x_.remove(coor[0])
#
#     elif coor[0] not in x_:
#         x_.append(coor[0])
#
#     if coor[1] in y_:
#         y_.remove(coor[1])
#     elif coor[1] not in y_:
#         y_.append(coor[1])
#
#
# x_.extend(y_)
#
# answer = x_

##############################

#  rectangle2
n = 5
m = 3
star = "*" *(n*m)
rectangle = "**\n**"
a = '**'+'\n'+'**'

rectangle = ""
for i in range(len(star)):
    rectangle +="*"
    if ((i+1)% 5)  == 0:
        rectangle += "\n"

print(rectangle)
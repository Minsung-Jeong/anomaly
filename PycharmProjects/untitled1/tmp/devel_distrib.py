import math

progresses = [40, 93, 30, 55, 60, 65]
speeds = [60, 1, 30, 5 , 10, 7]
full = 100

end_date = []

for i in range(len(progresses)):
    progre = progresses[i]
    speed = speeds[i]
    taking_tm = (full - progre)/speed

    end_date.append(math.ceil(taking_tm))

end_date.append(101)

distrib = []
answer = []

Many = 0
for i in range(len(end_date)-1):
    Many += 1
    # if (end_date[i] < end_date[i+1]) & (end_date[i+1] not in distrib):
    if (end_date[i] < end_date[i + 1]) & ( sum([x >= end_date[i+1] for x in distrib])==0 ):

        distrib.extend(end_date[:i])
        answer.append(Many)
        Many = 0


# distrib = [1,3,5]
# end_date = 4
#
#
#
# sum([x >= end_date[i+1] for x in distrib])
#     sum(a) == 0
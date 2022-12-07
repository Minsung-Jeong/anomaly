from datetime import timedelta
# fees = [기본시간, 기본요금, 단위시간, 단위요금]
# records = ["시각 차량번호 내역", "....."]

# records 의 최대길이는 1000
fees = [180, 5000, 10, 600]
records = ["05:34 5961 IN",
           "06:00 0000 IN",
           "06:34 0000 OUT",
           "07:59 5961 OUT",
           "07:59 0148 IN",
           "18:59 0000 IN", "19:09 0148 OUT", "22:59 5961 IN", "23:00 5961 OUT"]

record = records[0].split()
time  = record[0]
car = record[1]
stat = record[2]


cars = list(set([int(r.split()[1]) for r in records]))
cars.sort()


i = 0
temp = []
for r in records:
    print(r)
    if int(r.split()[1]) == cars[i]:
        temp.append(r.split()[0])
if len(temp) % 2 != 0:
    temp.append('23:59')

# 시간
r.split()[0][:2]
r.split()[0][3:]
# 분
temp

# 시간
int(temp[1][:2]) - int(temp[0][:2])
# 분
int(temp[1][3:]) - int(temp[0][3:])

# 시간
int(temp[3][:2]) - int(temp[2][:2])
# 분
int(temp[3][3:]) - int(temp[2][3:])

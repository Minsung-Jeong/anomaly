import pandas as pd


"""
1,000,000개의 데이터
ID : 샘플 고유 id
항공편 운항 관련 정보
Month: 해당 항공편의 출발 월
Day_of_Month: Month에 해당하는 월의 날짜
Estimated_Departure_Time: 전산 시스템을 바탕으로 측정된 비행기의 출발 시간 (현지 시각, HH:MM 형식)
Estimated_Arrival_Time: 전산 시스템을 바탕으로 측정된 비행기의 도착 시간 (현지 시각, HH:MM 형식)
Cancelled: 해당 항공편의 취소 여부 (0: 취소되지 않음, 1: 취소됨)
Diverted: 해당 항공편의 경유 여부 (0: 취소되지 않음, 1: 취소됨)
Origin_Airport: 해당 항공편 출발 공항의 고유 코드 (IATA 공항 코드) /// drop 가능
Origin_Airport_ID: 해당 항공편 출발 공항의 고유 ID (US DOT ID)
Origin_State: 해당 항공편 출발 공항이 위치한 주의 이름
Destination_Airport: 해당 항공편 도착 공항의 고유 코드 (IATA 공항 코드) /// drop 가능
Destination_Airport_ID: 해당 항공편 도착 공항의 고유 ID (US DOT ID)
Destination_State: 해당 항공편 도착 공항이 위치한 주의 이름
Distance: 출발 공항과 도착 공항 사이의 거리 (mile 단위)
Airline: 해당 항공편을 운항하는 항공사
Carrier_Code(IATA): 해당 항공편을 운항하는 항공사의 고유 코드 
(IATA 공항 코드, 단 다른 항공사가 같은 코드를 보유할 수도 있음)
Carrier_ID(DOT): 해당 항공편을 운항하는 항공사의 고유 ID (US DOT ID)
Tail_Number: 해당 항공편을 운항하는 항공기의 고유 등록번호
Delay: 항공편 지연 여부 (Not_Delayed, Delayed)
예측해야 하는 타깃
다수의 데이터에 레이블이 존재하지 않음
준지도학습을 통해 레이블링 가능
"""
df = pd.read_csv("C://data_minsung/dacon/airport/train.csv")


df.describe()
df.info()

def convert_cat(variable):
    temp = list(set(df[variable]))
    temp_dict = {temp[i]: i for i in range(len(temp))}
    for x in temp_dict:
        df[variable][df[variable] == x] = temp_dict[x]
    print('dict_len :',len(temp_dict))
    return df[variable].astype(float), temp_dict

def convert_cat_with_dict(variable, dict):
    temp_dict = dict
    for x in temp_dict:
        df[variable][df[variable] == x] = temp_dict[x]
    return df[variable].astype(float)

df['Origin_Airport'], air_dict = convert_cat('Origin_Airport')
len(air_dict)

df['Origin_State'], State_dict = convert_cat('Origin_State')
df['Destination_State'] = convert_cat_with_dict('Destination_State', State_dict)
df['Airline'], airline_dict = convert_cat('Airline')
df['Carrier_Code(IATA)'], airline_dict = convert_cat('Carrier_Code(IATA)')

df['Tail_Number'], airline_dict = convert_cat('Tail_Number')
df['Delay'], airline_dict = convert_cat('Delay')

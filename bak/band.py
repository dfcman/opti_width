#-*- coding:utf-8 -*-

import numpy as _numpy
import pandas as _pandas
from typing import (
    Tuple as _Tuple,
    List as _List,
)
from collections import Counter as _Counter
from tqdm import tqdm as _tqdm



## light 버전
def combinations(
    units: _List[int], 
    min_count: int=2,
    max_count: int=8,
    lower_bound: int=4500,
    upper_bound: int=4880,
) -> _List[_Tuple[int]]:
    print("####  가능한 조합 찾기  #####")
    result = []
    loss_list = []  # loss 리스트 
    result_app = result.append
    loss_list_app = loss_list.append
    counter_units = _Counter(units)  # Raw 카운터
    print("counter_units", counter_units)
    for i in _tqdm(range(max_count)):
        # 대상 리스트 정의
        units_i = [k for k,v in counter_units.items() for _ in range(v)]
        print("-"*10)
        print("----------------%d" %i)
        print("type(units_i)", type(units_i))
        print("units_i", units_i)
        n_units = len(units_i)
        print("n_unit", n_units)
        
        # 열이름 생성
        fmt = "width_%d"
        names = [fmt%(j) for j in range(i+1)]
        
        # 대상 유닛 생성
        df = _pandas.DataFrame({"key": [1]*n_units, fmt%(i): units_i})
        print(df)
    
        if i == 0:
            # 최초 Left를 정의한다
            left_df = df
        else:
            # Full Outer Merge를 수행한다.

            # Right를 정의한다.
            # Left의 최소 지폭보다 작거나 같아야 Merge를 수행할 수 있다.
            bound = upper_bound - _numpy.min(left_df[names[:i]].sum(axis=1))
            right_df = df.query("%s <= %d"%(fmt%(i), bound)).reset_index(drop=True)
            if not right_df.shape[0]:
                # Right에 더 이상 데이터 없어서 진행 불가다.
                break
            
            # 현재 left_df에서 지폭을 추가할 여유가 없는 경우를 삭제한다.
            min_unit = _numpy.min(units_i)
            filter = (upper_bound - left_df[names[:i]].sum(axis=1)).apply(lambda v:v>=min_unit)
            left_df = left_df[filter]
            
            # Full Outer Merge를 수행한다.
            left_df =_pandas.merge(left=left_df, right=right_df, on=["key"], how="outer")
            
            # Merge결과에 중복된 결과를 제거한다.
            left_df = left_df.drop_duplicates().reset_index(drop=True)
            
            # Merge결과에 최대 지폭을 넘는 경우를 삭제한다.
            filter = left_df[names].sum(axis=1).apply(lambda v:v<=upper_bound)
            left_df = left_df[filter]
            
            # Merge결과에 대해 열에 대한 오름차순 정렬한다.
            raws = []
            for raw in left_df.itertuples():
                raws.append([raw[1]]+sorted(raw[2:]))
            left_df = _pandas.DataFrame(raws, columns=left_df.columns)

            # Merge결과에 중복된 결과를 제거한다.
            left_df = left_df.drop_duplicates().reset_index(drop=True)

        if bool(left_df.shape[0]):
            # 결과가 존재한다면....
            if min_count <= (i+1):
                # 최소 그룹수보다 크거나 같다면, 
                # 최대/최소 범위 내에 있는 패턴만 꺼낸다.
                selector = left_df[names].sum(axis=1).apply(lambda v:v>=lower_bound)
                nck = left_df[selector]
                if not bool(nck.shape[0]):
                    continue
                for t in nck[names].itertuples():
                    result_app(tuple(t[1:]))
                    loss_list_app(upper_bound-sum(t[1:]))
                    
        else:
            # Left에 데이터가 없어서 더 이상 진행 불가다.
            break
        
        # 카운터 수정
        for k,v in counter_units.items():
            if v > 0:
                counter_units[k] -= 1
        
    return result, loss_list



# ## 모든 경우 고려하는 버전
# def combinations(
#     units: _List[int], 
#     min_count: int=2,
#     max_count: int=8,
#     lower_bound: int=4500,
#     upper_bound: int=4880,
# ) -> _List[_Tuple[int]]:
#     result = []
#     result_app = result.append
#     counter_units = _Counter(units)  # Raw 카운터
#     n_units = len(units)
#     for i in _tqdm(range(max_count)):
        
#         # 열이름 생성
#         fmt = "width_%d"
#         names = [fmt%(j) for j in range(i+1)]
        
#         # 대상 유닛 생성
#         df = _pandas.DataFrame({"key": [1]*n_units, fmt%(i): units})

#         if i == 0:
#             # 최초 Left를 정의한다
#             left_df = df
#         else:
#             # Full Outer Merge를 수행한다.

#             # Right를 정의한다.
#             # Left의 최소 지폭보다 작거나 같아야 Merge를 수행할 수 있다.
#             bound = upper_bound - _numpy.min(left_df[names[:i]].sum(axis=1))
#             right_df = df.query("%s <= %d"%(fmt%(i), bound)).reset_index(drop=True)
#             if not right_df.shape[0]:
#                 # Right에 더 이상 데이터 없어서 진행 불가다.
#                 break
            
#             # 현재 left_df에서 지폭을 추가할 여유가 없는 경우를 삭제한다.
#             min_unit = _numpy.min(units)
#             filter = (upper_bound - left_df[names[:i]].sum(axis=1)).apply(lambda v:v>=min_unit)
#             left_df = left_df[filter]
            
#             # Full Outer Merge를 수행한다.
#             left_df =_pandas.merge(left=left_df, right=right_df, on=["key"], how="outer")
            
#             # Merge결과에서 중복된 결과를 제거한다.
#             left_df = left_df.drop_duplicates().reset_index(drop=True)
            
#             # Merge결과에서 최대 지폭을 넘는 경우를 삭제한다.
#             filter = left_df[names].sum(axis=1).apply(lambda v:v<=upper_bound)
#             left_df = left_df[filter].reset_index(drop=True)
            
#             # Merge결과에서 raw_counter 개수에 안맞는 경우를 삭제한다.
#             left_df2 = left_df.drop(["key"], axis=1)
#             bool_series = _pandas.Series(True, index=range(len(left_df2)))
#             for i in range(len(left_df2)):
#                 counter_rows = _Counter(left_df2.loc[i])
#                 if not all([value<=counter_units[key] for key,value in counter_rows.items()]):
#                     bool_series[i] = False
#             left_df = left_df[filter].reset_index(drop=True)
            
#             # Merge결과에 대해 열에 대한 오름차순 정렬한다.
#             raws = []
#             for raw in left_df.itertuples():
#                 raws.append([raw[1]]+sorted(raw[2:]))
#             left_df = _pandas.DataFrame(raws, columns=left_df.columns)

#             # Merge결과에서 중복된 결과를 제거한다.
#             left_df = left_df.drop_duplicates().reset_index(drop=True)

#         if bool(left_df.shape[0]):
#             # 결과가 존재한다면....
#             if min_count <= (i+1):
#                 # 최소 그룹수보다 크거나 같다면, 
#                 # 최대/최소 범위 내에 있는 패턴만 꺼낸다.
#                 selector = left_df[names].sum(axis=1).apply(lambda v:v>=lower_bound)
#                 nck = left_df[selector]
#                 if not bool(nck.shape[0]):
#                     continue
#                 for t in nck[names].itertuples():
#                     # result.append(tuple(t[1:]))
#                     result_app(tuple(t[1:]))
#         else:
#             # Left에 데이터가 없어서 더 이상 진행 불가다.
#             break
        
        
#     return result


# i+=1


if __name__ == "__main__":
    
    lower_count = 2
    upper_count = 8
    lower_bound = 4500
    upper_bound = 4880
    orders = dict([  # test4
        (1000,10),
        (1050,17),
        (1150,19),
        (1200,15),
        (1250,13),
        (1300,24),
        (1350,15),
        (1400,13),
        (1450,12),
        (1500,28),
        (1550,15),
        (1600,48),
        (1650,91),
        (1700,40),
        (1750,47),
        (1800,62),
        (1850,32),
        (1900,92),
        (2000,45),
        (2050,25),
        (2100,37),
        (2200,37),
        (2300,19),
        (2400,56),
        (2500,34),
    ])
    
    units = [v[0] for v in orders.items()]  # 지폭 리스트
    counts = [v[1] for v in orders.items()]  # 지폭 별 롤 개수 리스트
    
    # 최소-최대 선택 가능 수량범위 계산
    max_selected = {}  # 지폭 별 최대 선택 개수 딕셔너리
    for unit in units:
        n_max = min(upper_bound // unit, upper_count)
        max_selected[unit] = {"max": n_max}
    sample_list = []
    n_maxs = []
    for u, v in max_selected.items():
        n_max = min(v["max"], orders[u])  # 실제 선택 가능한 최대 개수 (orders의 개수 고려)
        n_maxs.append(n_max)
        sample_list.extend([u]*n_max)
    
    # min_count 구하기
    max_unit = max(units)
    part = []
    flag_break = False
    for i in range(len(units)):
        for _ in range(n_maxs[i]):
            part += [sorted(units, reverse=True)[i]]
            if sum(part) > lower_bound:
                del part[-1]
                if lower_bound <= sum(part) <= upper_bound:
                    flag_break = True
                    break
        if flag_break:
            break
    if flag_break:
        min_count = len(part)  # 선택 가능한 최대 개수
    else:
        min_count = int(_numpy.ceil((lower_bound / max_unit)))
    
    # max_count 구하기
    min_unit = min(units)  # 가장 짧은 지폭
    min_unit_mc = n_maxs[0]  # min_unit 기준, 실제 선택 가능한 최대 개수
    if (upper_bound // min_unit) <= min_unit_mc:
        max_count = min_unit_mc
    else:
        part = []
        flag_break = False
        for i in range(len(units)):
            for _ in range(n_maxs[i]):
                part += [units[i]]
                if sum(part) > upper_bound:
                    del part[-1]
                    if lower_bound <= sum(part) <= upper_bound:
                        flag_break = True
                        break
            if flag_break:
                break
        if flag_break:
            max_count = len(part)  # 선택 가능한 최대 개수
        else:
            max_count = (upper_bound // min_unit)
            
    import time
    t1 = time.time()
    a = combinations(sample_list, min_count, max_count, lower_bound, upper_bound)
    print(f"\n연산시간 : {_numpy.round(time.time()-t1, 2)} sec")
    
    
    # from sys import getsizeof
    # getsizeof(a)

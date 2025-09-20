# -*- coding: utf-8 -*-
## Import ##
import numpy as np

#---- Function ----#
def preprocessing(order: dict,
                  standard_value: int,  # 묶을 대상 판단 기준 (미만 값)
                  n: int,  # 묶을 개수
                  flag_processing: bool=True):  # 전처리 사용여부
    """
    1000 미만인 지폭을 2개씩 묶기
    
    *** 전처리 필요한 오더 기준 (하나라도 만족 시) 
        1. 지폭 종류 35개 이상 & 850 미만 지폭 종류 5개 이상
        2. 지폭 종류 50개 이상
    """
    if flag_processing:
        if any([(len(order) >= 35)*(sum([1 for i in order.keys() if i<850]) >= 5) , (len(order) >= 50)]):
            print("\n===== 전처리 진행 =====\n")
            # standard_value 유효여부 판단 
            if standard_value >= min(order.keys())*n:
                raise
            
            order_pr = dict()
            preproc_dict = dict()  # 전처리 한 지폭 목록
            for key,value in order.items():
                if key >= standard_value:
                    order_pr[key] = order[key]
            
            for key,value in order.items():
                if key >= standard_value:
                    break
                if value%n==0:  # n의 배수라면
                    if key*n in order.keys():
                        order_pr[key*n] = order[key*n] + int(value/n)
                    else:
                        order_pr[key*n] = int(value/n)
                    preproc_dict[key] = int(value)
                else:  # n의 배수가 아니라면
                    if value < n:
                        order_pr[key] = value
                    else:  # 묶을 수 있을만큼 묶고 나머지 남기기
                        order_pr[key] = value % n
                        if key*n in order.keys():
                            order_pr[key*n] = order[key*n] + int(value/n)
                        else:
                            order_pr[key*n] = int(value/n)
                        preproc_dict[key] = int(value-(value % n))
        else:
            order_pr = order.copy()
            print("\n===== 전처리 진행2 =====\n")
            preproc_dict = dict()  # 전처리 한 지폭 목록
    
    else:
        order_pr = order.copy()
        print("\n===== 전처리 진행3 =====\n")
        preproc_dict = dict()  # 전처리 한 지폭 목록
                
    return order_pr, preproc_dict

# def preprocessing(order: dict,
#                   ):
#     """
#     1000 미만인 지폭을 2개씩 묶기 (지폭*2+1 로 묶기[표시를 위함])
    
#     *** 전처리 필요한 오더 기준 (하나라도 만족 시) 
#         1. 지폭 종류 35개 이상 & 850 미만 지폭 종류 5개 이상
#         2. 지폭 종류 50개 이상
#     """
    
#     if any([(len(order) >= 35)*(sum([1 for i in order.keys() if i<850]) >= 5) , (len(order) >= 50)]):
#         print("\n===== 전처리 진행 =====\n")
#         order_pr = dict()
#         preproc_dict = dict()  # 전처리 한 지폭 목록
#         for key,value in order.items():
#             if key >= 1000:
#                 order_pr[key] = order[key]
                
#         for key,value in order.items():
#             if key >= 1000:
#                 break
#             if value%2==0:  # 2의 배수라면
#                 order_pr[key*2+1] = int(value/2)
#                 preproc_dict[key] = int(value)
#             else:  # 2의 배수가 아니라면
#                 if value==1:
#                     order_pr[key] = 1
#                 else:  # 묶을 수 있을만큼 묶고 나머지 1 남기기
#                     order_pr[key] = 1
#                     order_pr[key*2+1] = int(value/2)
#                     preproc_dict[key] = int(value-1)
#     else:
#         order_pr = order.copy()
#         preproc_dict = dict()  # 전처리 한 지폭 목록
                
#     return order_pr, preproc_dict





def afterprocessing(raw_orders: dict,  # 원래 오더
                    preproc_dict: dict,  # 전처리 한 지폭 목록 (preprocessing 리턴 값)
                    n: int,  # 묶은 개수
                    result: np.array,  # optimize 결과 값
                    ):
    """
    전처리(1000 미만인 지폭을 2개씩 묶기) 후 도출된 결과를 다시 되돌리기
    """
    print("\n===== 후처리 진행 =====\n")
    result_list = [list(map(int, list(list(result)[i]))) for i in range(len(result))]  # 요소 int화 & 리스트 화
    result_list = [[result_list[i][m] for m in range(8) if not result_list[i][m]==0] for i in range(len(result_list))]  # 0 제거
    
    for key,value in preproc_dict.items():
        ele_proc = int(key*n)  # 후처리 대상 지폭
        n_proc = int(value/n)  # 남은 처리 횟수
        # print(ele_proc, n_proc)
        flag_break = False  # 처리 중단 여부
        i = 0
        while not flag_break:
            if ele_proc in result_list[i]:  # 되돌릴 요소가 있으면
                count = result_list[i].count(ele_proc)  # 보유 개수
                # print("처리할 것 있음", count)
                if count <= n_proc:
                    n_iter = count
                else:
                    n_iter = n_proc
                for _ in range(n_iter):
                    del_index = result_list[i].index(ele_proc)  # 인덱스 서치
                    del result_list[i][del_index]
                    n_proc -= 1  # 남은 처리 횟수 -1
                    result_list[i].extend([key]*n)
                    # print("--n_proc: ", n_proc)
            
            if n_proc == 0:  # while문 break 판단
                flag_break = True
            i += 1
    
    afterproc_order = {}
    for i in range(len(result_list)):
        for m in range(len(result_list[i])):
            if result_list[i][m] in afterproc_order.keys():
                afterproc_order[result_list[i][m]] += 1
            else:
                afterproc_order[result_list[i][m]] = 1
    
    # 정상 후처리 여부 판단
    if afterproc_order == raw_orders:
        print("\n후처리 완료")
    else:
        raise 
    
    # np.array 화
    result_array = np.zeros((len(result_list),8))  # 결과 행렬
    for i in range(len(result_list)):
        for m in range(len(result_list[i])):
            result_array[i][m] = result_list[i][m]
            
    
    return result_array

# def afterprocessing(raw_orders: dict,  # 원래 오더
#                     preproc_dict: dict,  # 전처리 한 지폭 목록 (preprocessing 리턴 값)
#                     result: np.array,  # optimize 결과 값
#                     ):
#     """
#     전처리(1000 미만인 지폭을 2개씩 묶기) 후 도출된 결과를 다시 되돌리기 (지폭*2+1 되돌리기)
#     """
#     print("\n===== 후처리 진행 =====\n")
#     result_list = [list(map(int, list(list(result)[i]))) for i in range(len(result))]  # 요소 int화 & 리스트 화
#     result_list = [[result_list[i][m] for m in range(8) if not result_list[i][m]==0] for i in range(len(result_list))]  # 0 제거
    
#     for key,value in preproc_dict.items():
#         ele_proc = int(key*2+1)  # 후처리 대상 지폭
#         n_proc = int(value/2)  # 남은 처리 횟수
#         # print(ele_proc, n_proc)
#         flag_break = False  # 처리 중단 여부
#         i = 0
#         while not flag_break:
#             if ele_proc in result_list[i]:  # 되돌릴 요소가 있으면
#                 count = result_list[i].count(ele_proc)  # 보유 개수
#                 # print("처리할 것 있음", count)
#                 if count <= n_proc:
#                     n_iter = count
#                 else:
#                     n_iter = n_proc
#                 for _ in range(n_iter):
#                     del_index = result_list[i].index(ele_proc)  # 인덱스 서치
#                     del result_list[i][del_index]
#                     n_proc -= 1  # 남은 처리 횟수 -1
#                     result_list[i].extend([key]*2)
#                     # print("--n_proc: ", n_proc)
            
#             if n_proc == 0:  # while문 break 판단
#                 flag_break = True
#             i += 1
    
#     afterproc_order = {}
#     for i in range(len(result_list)):
#         for m in range(len(result_list[i])):
#             if result_list[i][m] in afterproc_order.keys():
#                 afterproc_order[result_list[i][m]] += 1
#             else:
#                 afterproc_order[result_list[i][m]] = 1
    
#     # 정상 후처리 여부 판단
#     if afterproc_order == raw_orders:
#         print("\n후처리 완료")
#     else:
#         raise 
    
#     # np.array 화
#     result_array = np.zeros((len(result_list),8))  # 결과 행렬
#     for i in range(len(result_list)):
#         for m in range(len(result_list[i])):
#             result_array[i][m] = result_list[i][m]
            
    
#     return result_array



if __name__ == "__main__":
    order_pr, preproc_dict = preprocessing(raw_orders)
    result_array = afterprocessing(orders,
                                    preproc_dict,
                                    result)
    
    len(np.unique(result, axis=0))

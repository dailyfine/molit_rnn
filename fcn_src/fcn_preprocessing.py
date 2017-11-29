# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import functools

max_time_index = 50
batch_size = 0

molit_start_time = '201601'
molit_end_time = '201704'

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def disjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

def molit_timecode_generator(start_code, end_code):
    result_timecode = []

    start_YY = int(start_code[:4])
    start_MM = int(start_code[4:])

    end_YY = int(end_code[:4])
    end_MM = int(end_code[4:])

    tmp_MM = start_MM
    tmp_YY = start_YY

    while tmp_YY * 100 + tmp_MM < end_YY * 100 + end_MM:
        temp_time = tmp_MM + tmp_YY * 100
        result_timecode.append(str(temp_time) + '_molit.csv')
        tmp_MM = tmp_MM + 1

        if tmp_MM >= 13:
            tmp_MM = 1
            tmp_YY = tmp_YY + 1

    return result_timecode

def ggi_timecode_generator(start_code, end_code):
    result_timecode = []

    start_YY = int(start_code[:4])
    start_MM = int(start_code[4:])

    end_YY = int(end_code[:4])
    end_MM = int(end_code[4:])

    tmp_MM = start_MM
    tmp_YY = start_YY

    while tmp_YY * 100 + tmp_MM < end_YY * 100 + end_MM:
        temp_time = tmp_MM + tmp_YY * 100
        result_timecode.append(str(temp_time) + '_ggi.csv')
        tmp_MM = tmp_MM + 1

        if tmp_MM >= 13:
            tmp_MM = 1
            tmp_YY = tmp_YY + 1

    return result_timecode

def fcn_access_disk(molit_time_code ='201612'):

    encoder_input = []
    encoder_target = []
    item = molit_time_code
    data_molit = pd.read_csv('data_files/molit/' + item + '_molit.csv')
    # 레코드가 시/도/동 까지 같은 경우 지번이나 아프명 중에 어느 하나라도 일치하면 같은 매물로 여긴다.
    raw_input = data_molit[['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']]
    raw_input_batch = raw_input.as_matrix()

    for item in raw_input_batch:
        if isConvertible_to_float(item):
            encoder_input.append(item[:-1])
            encoder_target.append(item[-1])
    return encoder_input, encoder_target

def isConvertible_to_float(record):
    for i in record:
        try:
            np.asarray(i, np.float32)
        except:
            print("Error! it's not convertible to float!")
            return False
    return True

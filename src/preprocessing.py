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

def sequence_generator(auction_time_code = '201612'):
    print('data_files/ggi/' + auction_time_code)
    data_ggi = pd.read_csv('data_files/ggi/' + auction_time_code + '_ggi.csv')

    # Dealing with nan
    # STEP1) remove columns with high probability of nan.
    data_ggi = data_ggi.drop('유치권여부', 1)
    data_ggi = data_ggi.drop('법정지상권여부', 1)
    data_ggi = data_ggi.drop('지분여부', 1)
    data_ggi = data_ggi.drop('아파트평형', 1)

    # STEP2) Filling NA which is not that important.

    data_ggi['전용면적'].fillna(81)
    data_ggi['대지면적'].fillna(44)
    data_ggi['층'].fillna(1)

    # STEP3) If very important information is nan, then we will ignore that record.
    data_ggi = data_ggi.dropna(axis=0)
    encoder_batch = []
    decoder_batch = []

    for idx in range (2):         #(data_ggi.shape[0]): #나중에 이렇게 고쳐야함
        province = data_ggi.iloc[idx]['시도']
        city = data_ggi.iloc[idx]['시구군']
        village = data_ggi.iloc[idx]['읍면동']
        number = data_ggi.iloc[idx]['지번']
        name = data_ggi.iloc[idx]['아파트명']
        size = data_ggi.iloc[idx]['전용면적']

        result = pd.DataFrame()
        for item in molit_timecode_generator(molit_start_time, molit_end_time):
            # for item in molit_timecode_generator('201101', auction_time_code):
            # 이게 더 효율일지는 나중에 체크해 보도록 하자.
            data_molit = pd.read_csv('data_files/molit/' + item)
            c1 = data_molit['시도'] == province
            c2 = data_molit['시구군'] == city
            c3 = data_molit['읍면동'] == village
            c4 = data_molit['지번'] == number
            c5 = data_molit['아파트명'] == name
            c6 = data_molit['전용면적'] < (size + 10)
            c7 = data_molit['전용면적'] > (size - 10)

            # 레코드가 시/도/동 까지 같은 경우 지번이나 아프명 중에 어느 하나라도 일치하면 같은 매물로 여긴다.
            data_filtered = pd.DataFrame(data_molit[conjunction(c1, c2, c3, disjunction(c4, c5), c6 ,c7)])
            result = result.append(data_filtered)

        cleaned_encoder_batch_ = result[['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']]
        cleaned_encoder_batch_element = cleaned_encoder_batch_.dropna(axis=0).as_matrix()
        cleaned_decoder_batch_element = data_ggi.iloc[idx][['위도', '경도','전용면적', '대지면적', '층', '낙찰년도', '낙찰월', '낙찰가']].as_matrix()

        if not result.empty:

            if isConvertible_to_float(cleaned_encoder_batch_element) and isConvertible_to_float(cleaned_decoder_batch_element):
                encoder_batch.append(cleaned_encoder_batch_element)
                decoder_batch.append(cleaned_decoder_batch_element)

            else:
                print("we skipped this since it's not convertible to float.")
                print(province, city, village, number, name)
                print("------------------------------------------------------")

    return encoder_batch, decoder_batch

# def make_prebatch(start_code='201501', end_code='201502'):
#     encoder_batch = []
#     decoder_batch = []
#
#     tmp_encoder_batch, tmp_decoder_batch = sequence_generator(start_code+'_ggi.csv')
#     encoder_batch = encoder_batch + tmp_encoder_batch
#     decoder_batch = decoder_batch + tmp_decoder_batch
#
#     return encoder_batch, decoder_batch

# encoder_batch 를 np matrix로 바꾸더라도 (a,b,c)와 같은 shape이 아니라 (a,)
# 로만 나온다. 왜냐하면 현재 b의 크기(time_step)이 유동적으로 계속 변하는 값이기 때문에 그렇다.
# 반면 decoder_batch는 이렇게 하면 자연스럽게 (a,c) 로 나온다.
# 이때 a = 배치사이즈, b = time_step, c = input_feature size 이다.

def make_variable_length_batch(timecode='201612'):

    e, c = sequence_generator(timecode)
    c = np.asarray(c)
    final_encoder_input_batch = []
    final_encoder_output_batch = []
    final_decoder_input_batch = []
    final_decoder_output_batch = []

    for batch_idx, element_in_the_batch in enumerate(e):
        encoder_input_batch  = []
        encoder_output_batch = []

        for idx, item in enumerate(element_in_the_batch):
        #  ['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']

            if idx < np.shape(element_in_the_batch)[0] - 1:
                latitude = element_in_the_batch[idx][0]
                longitude = element_in_the_batch[idx][1]
                current_apartment_selling_year = element_in_the_batch[idx][2]
                current_apartment_selling_month = element_in_the_batch[idx][3]
                current_apartment_constructed_year = element_in_the_batch[idx][4]
                current_apartment_floor = element_in_the_batch[idx][5]
                current_apartment_size = element_in_the_batch[idx][6]
                current_apartment_selling_price = element_in_the_batch[idx][7]

                predicting_apartment_selling_year = element_in_the_batch[idx + 1][2]
                predicting_apartment_selling_month = element_in_the_batch[idx + 1][3]
                predicting_apartment_floor = element_in_the_batch[idx + 1][5]
                predicting_apartment_size = element_in_the_batch[idx + 1][6]

                predicting_apartment_selling_price = element_in_the_batch[idx + 1][7]

                input_batch_pre =\
                    [ latitude
                    , longitude
                    , current_apartment_selling_year
                    , current_apartment_selling_month
                    , current_apartment_floor
                    , current_apartment_size
                    , current_apartment_selling_price
                    , current_apartment_constructed_year
                    , predicting_apartment_selling_year
                    , predicting_apartment_selling_month
                    , predicting_apartment_floor
                    , predicting_apartment_size]

                encoder_input_batch.append(np.asarray(input_batch_pre))
                # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.
                encoder_output_batch.append(predicting_apartment_selling_price)

        if len(encoder_input_batch) != 0:
            final_encoder_input_batch.append(encoder_input_batch)
            final_encoder_output_batch.append(encoder_output_batch)
            final_decoder_input_batch.append(c[batch_idx][:-1])
            final_decoder_output_batch.append(c[batch_idx][-1])

    return final_encoder_input_batch, final_encoder_output_batch, final_decoder_input_batch, final_decoder_output_batch

def isConvertible_to_float(record):
    for i in record:
        try:
            np.asarray(i, np.float32)
        except:
            print("Error! it's not convertible to float!")
            return False
    return True

import h5py
import numpy as np
max_length = 50

def timecode_generator(start_code, end_code):
    result_timecode = []

    start_YY = int(start_code[:4])
    start_MM = int(start_code[4:])

    end_YY = int(end_code[:4])
    end_MM = int(end_code[4:])

    tmp_MM = start_MM
    tmp_YY = start_YY

    while tmp_YY * 100 + tmp_MM < end_YY * 100 + end_MM:
        temp_time = tmp_MM + tmp_YY * 100
        result_timecode.append(str(temp_time))
        tmp_MM = tmp_MM + 1

        if tmp_MM >= 13:
            tmp_MM = 1
            tmp_YY = tmp_YY + 1

    return result_timecode

t = timecode_generator('201501','201604')

for time_string in t:
    input_file_name = "fcn_batch_data_h5py/" + time_string + "_np"
    print("normalizing encoder_input in file ",input_file_name)
    h5f = h5py.File(input_file_name, 'r+')

    encoder_input_batch = h5f['fcn_input_batch'][:]
    encoder_target_batch = h5f['fcn_target_batch'][:]

    batch_size_ = len(encoder_input_batch)    # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.
    for i in range(batch_size_):
        # 201701년 기준으로 평균 및 표준 편차로 나누어줌
        # t[['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']]
        encoder_input_batch[i][0] = (encoder_input_batch[i][0] - 36.4367) / 1.0547  # 위도
        encoder_input_batch[i][1] = (encoder_input_batch[i][1] - 127.5861) / 0.8790  # 경도
        encoder_input_batch[i][2] = (encoder_input_batch[i][2] - 2014) / 2  # 거래년도 2011 - 2016 까지 평균 및 표준 편차
        encoder_input_batch[i][3] = (encoder_input_batch[i][3] - 6.5) / 3.45  # 거래월 2011 - 2016 까지 평균
        encoder_input_batch[i][4] = (encoder_input_batch[i][4] - 2000.1287) / 8.2695  # 건축년도
        encoder_input_batch[i][5] = (encoder_input_batch[i][5] - 8.659) / 5.7851  # 층
        encoder_input_batch[i][6] = (encoder_input_batch[i][6] - 72.7554) / 24.3750  # 면적
        encoder_target_batch[i] = (encoder_target_batch[i]- 223586273) / 122304694  # 거래가격

    h5f['fcn_input_batch'][:] = encoder_input_batch
    h5f['fcn_target_batch'][:] = encoder_target_batch
    h5f.close()
    print("Normalizing ", input_file_name, "completed!")

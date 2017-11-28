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

t = timecode_generator('201610', '201611')

for time_string in t:
    input_file_name = "batch_data_h5py/" + time_string + "_np"
    print("normalizing encoder_input in file ",input_file_name)
    h5f = h5py.File(input_file_name, 'r+')
    batch_size_ = len(h5f['encoder_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)
    encoder_input_batch = h5f['encoder_input_batch'][:]
    encoder_length_vec = h5f['encoder_length_vector_batch'][:]
    encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:], dtype=np.int)

    # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.
    for i in range(batch_size_):
        for j in range(max_length):
            # 201701년 기준으로 평균 및 표준 편차로 나누어줌
            encoder_input_batch[i][j][0] = (encoder_input_batch[i][j][0] - 36.4367) / 1.0547  # 위도
            encoder_input_batch[i][j][1] = (encoder_input_batch[i][j][1] - 127.5861) / 0.8790  # 경도
            encoder_input_batch[i][j][2] = (encoder_input_batch[i][j][2] - 2014) / 2  # 거래년도 2011 - 2016 까지 평균 및 표준 편차
            encoder_input_batch[i][j][3] = (encoder_input_batch[i][j][3] - 6.5) / 3.45  # 거래월 2011 - 2016 까지 평균
            encoder_input_batch[i][j][4] = (encoder_input_batch[i][j][4] - 8.659) / 5.7851  # 층
            encoder_input_batch[i][j][5] = (encoder_input_batch[i][j][4] - 72.7554) / 24.3750  # 면적
            encoder_input_batch[i][j][6] = (encoder_input_batch[i][j][6] - 223586273) / 122304694  # 거래가격
            encoder_input_batch[i][j][7] = (encoder_input_batch[i][j][7] - 2000.1287) / 8.2695  # 건축년도
            encoder_input_batch[i][j][8] = (encoder_input_batch[i][j][8] - 2014) / 2  # 예측년도 2011 - 2016 까지 평균 및 표준 편차
            encoder_input_batch[i][j][9] = (encoder_input_batch[i][j][9] - 6.5) / 3.45  # 예측월 2011 - 2016 까지 평균
            encoder_input_batch[i][j][10] = (encoder_input_batch[i][j][10] - 8.659) / 5.7851  # 예측 층
            encoder_input_batch[i][j][11] = (encoder_input_batch[i][j][11] - 72.7554) / 24.3750  # 면적

    encoder_target_batch = (h5f['encoder_target_batch'][:] - 223586273) / 122304694
    h5f['encoder_input_batch'][:] = encoder_input_batch
    h5f['encoder_target_batch'][:] = encoder_target_batch
    h5f.close()
    print("Normalizing ", input_file_name, "completed!")

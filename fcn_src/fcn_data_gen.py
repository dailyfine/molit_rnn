import fcn_src.fcn_preprocessing as preprocessing
import h5py
import numpy as np
import time

max_length = 50
feature_number = 12

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

# 고정된 사이즈로 encoder input, output 배치를 만든다.

def writeh5py(timecode):
    print(range(len(timecode)))
    for i in range(len(timecode)):
        print('\n')
        print(timecode[i])
        print("Starting time:", time.ctime())
        encoder_input_batch, encoder_target_batch = preprocessing.fcn_access_disk(timecode[i])

        encoder_input_batch = np.asarray(encoder_input_batch,dtype=np.float32)
        encoder_target_batch = np.asarray(encoder_target_batch,dtype=np.float32)

        h5f = h5py.File('fcn_batch_data_h5py/' + timecode[i] + '_np', 'w')
        h5f.create_dataset('fcn_input_batch', data=encoder_input_batch)
        h5f.create_dataset('fcn_target_batch', data=encoder_target_batch)
        h5f.create_dataset('fcn_batch_length', data=len(encoder_input_batch))
        h5f.close()
        print("Finishing time:", time.ctime())
        print("Saving completed at", timecode[i])
        print("data encoder size is", str(len(encoder_input_batch)))
        print("\n")

writeh5py(timecode_generator('201501','201604'))

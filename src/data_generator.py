import src.preprocessing as preprocessing
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
def make_batch(encoder_input, encoder_output):
    encoder_input_batch = []
    encoder_target_batch = []

    for batch_idx, ith_input_batch in enumerate(encoder_input):
        input_ = encoder_input[batch_idx]
        target_ = encoder_output[batch_idx]
        t_in = np.asarray(input_)
        t_out = np.asarray(target_)

        z_in = np.zeros_like(np.arange(max_length * feature_number).reshape(max_length, feature_number),
                             dtype=np.float32)
        z_out = np.zeros_like(np.arange(max_length), dtype=np.float32)

        if t_in.shape[0] <= max_length:
            z_in[:t_in.shape[0], :t_in.shape[1]] = t_in
            z_out[:t_out.shape[0]] = t_out
        else:
            z_in = t_in[-max_length:]
            z_out = t_out[-max_length:]

        encoder_input_batch.append(z_in)
        encoder_target_batch.append(z_out)

    return np.asarray(encoder_input_batch), np.asarray(encoder_target_batch)


def writeh5py(timecode):
    print(range(len(timecode)))
    for i in range(len(timecode)):
        print('\n')
        print(timecode[i])
        print("Starting time:", time.ctime())

        ei, eo, di, do = preprocessing.make_variable_length_batch(timecode[i])

        ei_len_vec = np.zeros(len(ei))
        for idx,item in enumerate(ei):
            if np.shape(item)[0] > 50:
                ei_len_vec[idx] = 50
            else:
                ei_len_vec[idx] = np.shape(item)[0]

        encoder_input_batch, encoder_target_batch = make_batch(ei, eo)
        encoder_input_batch = np.asarray(encoder_input_batch,dtype=np.float32)
        encoder_target_batch = np.asarray(encoder_target_batch,dtype=np.float32)

        di = np.asarray(di,dtype=np.float32)
        do = np.asarray(do,dtype=np.float32)
        h5f = h5py.File('batch_data_h5py/' + timecode[i] + '_np', 'w')
        h5f.create_dataset('encoder_input_batch', data=encoder_input_batch)
        h5f.create_dataset('encoder_target_batch', data=encoder_target_batch)
        h5f.create_dataset('decoder_input_batch', data=di)
        h5f.create_dataset('decoder_target_batch', data=do)
        h5f.create_dataset('encoder_length_vector_batch', data=ei_len_vec)
        h5f.create_dataset('batch_length', data=len(ei))
        h5f.close()
        print(ei_len_vec)
        print("Finishing time:", time.ctime())
        print("Saving completed at", timecode[i])
        print("data encoder size is", str(len(ei)))
        print("\n")

writeh5py(timecode_generator('201501','201604'))

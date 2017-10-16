import preprocessing
import h5py
import numpy as np
model_path = "./tmp/hsmodel_new"
max_length = 300
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
        print(timecode[i])
        ei, eo, decoder_input_batch, decoder_target_batch = preprocessing.make_variable_length_batch(timecode[i])
        encoder_input_batch, encoder_target_batch = make_batch(ei, eo)
        encoder_input_batch = np.asarray(encoder_input_batch,dtype=np.float32)
        encoder_target_batch = np.asarray(encoder_target_batch,dtype=np.float32)
        decoder_input_batch = np.asarray(decoder_input_batch,dtype=np.float32)
        decoder_target_batch = np.asarray(decoder_target_batch,dtype=np.float32)

        h5f = h5py.File('batch_data_h5py/' + timecode[i] + '_np', 'w')
        h5f.create_dataset('encoder_input_batch', data=encoder_input_batch)
        h5f.create_dataset('encoder_target_batch', data=encoder_target_batch)
        h5f.create_dataset('decoder_input_batch', data=decoder_input_batch)
        h5f.create_dataset('decoder_target_batch', data=decoder_target_batch)
        h5f.close()
        print("Saving completed!")

writeh5py(timecode_generator('201501','201604'))

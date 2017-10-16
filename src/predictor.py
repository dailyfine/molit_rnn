import tensorflow as tf
import numpy as np
import h5py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = "./tmp/model_test_"
max_length = 300
feature_number = 12

Training_flag = False
Test_flag = True

restore_session_flag = True
time_start = '201701'
time_end = '201604'

learning_rate = 0.000001
n_hidden = 10
n_hidden2 = 128
n_hidden3 = 128

n_hidden_d = 30
n_hidden2_d = 30
n_hidden3_d = 50
h_hidden_sandwich_d = 100

n_epoch = 100
n_input = feature_number
n_mini_batch_size = 25
n_decoder_input_feature = 10


# length returns lenX such that
# lenX[idx_specifies_batch] = the true length of times_steps
# i.e., maxlength - zero vectors which is filled in a make_batch method.
# This length info is very useful when calculating the cost removing the effect of zero vector's output.

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length_ = tf.reduce_sum(used, 1)
    length_ = tf.cast(length_, tf.int32)
    return length_


# np version of above method.
def np_length(sequence):
    used = np.sign(np.amax(np.abs(sequence), 2))
    length_ = np.sum(used, 1)
    return length_


# this make_batch method fills the input_batch with zero_vectors to make a fixed size tim_step batch element.

def make_batch(encoder_input, encoder_output):
    encoder_input_batch = []
    encoder_target_batch = []

    for batch_idx, ith_input_batch in enumerate(encoder_input):
        print(batch_idx)
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
            z_in = t_in[0:max_length]
            z_out = t_out[0:max_length]

        encoder_input_batch.append(z_in)
        encoder_target_batch.append(z_out)

    return np.asarray(encoder_input_batch), np.asarray(encoder_target_batch)


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


# declare variables which will be defined in the for loop

pseudo_decoder_target_batch = np.arange(1)
decoder_input_batch = np.arange(1)
encoder_target_batch = np.arange(1)
encoder_input_batch = np.arange(1)
shuffled_batch_idx = np.arange(1)

cost = tf.zeros_like(n_mini_batch_size)

with tf.device('/device:cpu:0'):
    # Declare the model variable and place holder.
    X = tf.placeholder(tf.float32, [None, max_length, n_input])
    Y = tf.placeholder(tf.float32, [None, max_length])
    Z = tf.placeholder(tf.float32, [None, n_decoder_input_feature])
    Q = tf.placeholder(tf.float32, [None])
    lenXpl = tf.placeholder(tf.int32, [None])

    # initialization with reasonable scales. - this scale is optimized through experiments.
    W1 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), name="W1")
    b1 = tf.Variable(tf.random_normal([n_hidden2]), name="b1")

    W3 = tf.Variable(10*tf.random_normal([n_hidden2, 1]), name="W3")
    b3 = tf.Variable(10*tf.random_normal([]), name="b3")

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # dropout is unnecessary.
    # cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=length(X))

    W_d1 = 2*tf.Variable(tf.eye(n_hidden + n_decoder_input_feature, n_hidden2_d), name="W_d1")
    b_d1 = tf.Variable(tf.zeros([n_hidden2_d]), name="b_d1")

    W_d2 = tf.Variable(tf.eye(n_hidden2_d, h_hidden_sandwich_d), name="W_d2")
    b_d2 = tf.Variable(tf.zeros([h_hidden_sandwich_d]), name="b_d1")

    W_d3 = tf.Variable(tf.eye(h_hidden_sandwich_d, 1), name="W_d3")
    b_d3 = tf.Variable(tf.zeros([]), name="b_d3")

    # cost calculating using the tensors.

    encoder_cost_list = []
    decoder_cost_list = []

    encoder_result = []
    decoder_result = []

    for b_idx in range(n_mini_batch_size):
        #  encoder has 2 layers.
        encoder_layer1_output = tf.matmul(outputs[b_idx],
                                          W1) + b1
        encoder_layer1_output = tf.nn.relu(encoder_layer1_output)
        encoder_layer2_output = tf.matmul(encoder_layer1_output, W3) + b3

        true_last_hidden_state = outputs[b_idx][lenXpl[b_idx] - 1] * 0.01

        #  calculating encoder cost.
        z = np.zeros(max_length)  # same as np.zeros_like(max_length)
        len_ = lenXpl[b_idx]
        ones = tf.ones([len_], dtype=tf.float32) / tf.cast(len_, tf.float32)
        zeros = tf.zeros([max_length - len_])
        onezeros = tf.concat([ones, zeros], axis=0)

        meaningful_result = tf.abs(encoder_layer2_output - tf.reshape(Y[b_idx], [tf.shape(Y[b_idx])[0], 1]))
        meaningful_result = tf.multiply(onezeros, meaningful_result)
        encoder_cost_list.append(tf.reduce_sum(meaningful_result))
        encoder_result.append(encoder_layer2_output)  # last time_step encoder prediction result to print

        #  decoder has 2 layers.
        decoder_layer1_input = tf.reshape(tf.concat([true_last_hidden_state, Z[b_idx]], axis=0),
                                          [1, n_hidden + n_decoder_input_feature])
        decoder_layer1_output = tf.matmul(decoder_layer1_input, W_d1) + b_d1
        decoder_layer1_output = tf.nn.relu(decoder_layer1_output)

        decoder_layer_sandwich_output = tf.matmul(decoder_layer1_output, W_d2) + b_d2
        decoder_layer_sandwich_output = tf.nn.relu(decoder_layer_sandwich_output)
        decoder_layer2_output = tf.matmul(decoder_layer_sandwich_output, W_d3) + b_d3

        #  calculating decoder cost.
        decoder_cost_list.append(tf.abs(decoder_layer2_output - Q[b_idx]))
        decoder_result.append(decoder_layer2_output)

    encoder_result_tf = tf.stack(encoder_result)
    decoder_result_tf = tf.stack(decoder_result)  # decoder prediction result to print

    cost_ = tf.stack(encoder_cost_list)
    cost_decoder = tf.stack(decoder_cost_list)

    cost_result = tf.reduce_mean(tf.squeeze(cost_decoder)) + 0.00**tf.reduce_mean(cost_)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_result*1000)

    # run the session to calculate the result.

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if restore_session_flag:
    try:
        print("successfully restored the previous session!")
        saver.restore(sess, model_path)
    except:
        print("There is no stored model. I'm creating one at ->", model_path)

# declare variables.

t = timecode_generator('201501', '201702')
randint = np.random.randint(0, len(t))  # to select random mini-batch.

if Training_flag:

    for epoch in range(n_epoch):
        if epoch % 2 == 0:
            randint = np.random.randint(0, len(t)) # to select another random mini-batch in timescale.
            np.random.shuffle(shuffled_batch_idx) # to select another random mini-batch element in specific time data.

        h5f = h5py.File('batch_data_h5py/' + t[randint] + '_np', 'r')

        # If you wan to see the file where mini-batch is extracted. commenting out below.
        print("mini-batch has been drawn from " + 'batch_data_h5py/' + t[randint] + '_np')
        
        batch_size_ = len(h5f['encoder_input_batch'][:])
        shuffled_batch_idx = np.arange(batch_size_)

        encoder_input_batch = h5f['encoder_input_batch'][:]
        encoder_target_batch = h5f['encoder_target_batch'][:]
        decoder_input_batch = h5f['decoder_input_batch'][:]
        pseudo_decoder_target_batch = h5f['decoder_input_batch'][:] # This is used for initialzation.
        decoder_target_batch = h5f['decoder_target_batch'][:]

        h5f.close()
        lenX = np.asarray(np_length(encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]), dtype=np.int)

        np.random.shuffle(shuffled_batch_idx)
        _, loss = sess.run([optimizer, cost_result],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                               , Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                               , Z: decoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                               , Q: decoder_target_batch[shuffled_batch_idx][:n_mini_batch_size] #[:, 5]
                               , lenXpl: np.asarray(
                                   np_length(encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]), dtype=np.int)
                                      }
                           )
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    print('Optimization Complete!')
    s = saving_path = saver.save(sess, model_path)
    print('model saving completed!')

if Test_flag:

    h5f = h5py.File('batch_data_h5py/' + t[randint] + '_np', 'r')

    # If you wan to see the file where mini-batch is extracted. commenting out below.
    # print("mini-batch has been drawn from " + 'batch_data_h5py/' + t[randint] + '_np')

    batch_size_ = len(h5f['encoder_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)

    encoder_input_batch = h5f['encoder_input_batch'][:]
    encoder_target_batch = h5f['encoder_target_batch'][:]
    decoder_input_batch = h5f['decoder_input_batch'][:]
    pseudo_decoder_target_batch = h5f['decoder_input_batch'][:]
    decoder_target_batch = h5f['decoder_target_batch'][:]

    h5f.close()
    lenX = np.asarray(np_length(encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]), dtype=np.int)

    encoder_prediction_, decoder_prediction_ = sess.run([encoder_result_tf, decoder_result_tf],
                                                        feed_dict={X: encoder_input_batch[shuffled_batch_idx][
                                                                      :n_mini_batch_size]
                                                            , Y: encoder_target_batch[shuffled_batch_idx][
                                                                 :n_mini_batch_size]
                                                            , Z: decoder_input_batch[shuffled_batch_idx][
                                                                 :n_mini_batch_size]
                                                            , Q: decoder_target_batch[shuffled_batch_idx][
                                                                 :n_mini_batch_size]
                                                            , lenXpl: np.asarray(np_length(
                                                                encoder_input_batch[shuffled_batch_idx][
                                                                :n_mini_batch_size]), dtype=np.int)
                                                                   }
                                                        )
    encoder_result = []
    re = encoder_prediction_[:, :, 0]
    lenX = np.asarray(np_length(encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]), dtype=np.int)
    for i in range(n_mini_batch_size):
        tmp = [re[i][lenX[i] - 1], encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size][i][lenX[i] - 1]]
        encoder_result.append(tmp)
    result_np = np.asarray(encoder_result, dtype=np.float32)

    print(result_np)
    cost_of_test_result = np.mean(abs(result_np[:, 0] - result_np[:, 1]))

    print("encoder total cost: ", cost_of_test_result)
    decoder_result = []
    re = decoder_prediction_[:, :, 0]

    for i in range(n_mini_batch_size):
        tmp = [re[i], decoder_target_batch[shuffled_batch_idx][:n_mini_batch_size][i]]
        decoder_result.append(tmp)

    result_np = np.asarray(decoder_result, dtype=np.float32)
    print(['prediction ', ' target '])
    print(result_np)
    cost_of_test_result = np.mean(abs(result_np[:, 0] - result_np[:, 1]))

    print("decoder total cost: ", cost_of_test_result)


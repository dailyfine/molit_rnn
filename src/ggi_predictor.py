import tensorflow as tf
import numpy as np
import h5py
import os
model_path = "./tmp/model_test_"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_length = 50
feature_number = 12

restore_session_flag = True
Training_flag = False
Test_flag = True
learning_rate = 0.01
n_hidden = 10
n_hidden2 = 128
n_hidden3 = 128

n_epoch = 500
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

cost = tf.zeros_like(n_mini_batch_size)
with tf.device('/device:cpu:0'):
    # Declare the model variable and place holder.
    X = tf.placeholder(tf.float32, [None, max_length, n_input])
    Y = tf.placeholder(tf.float32, [None, max_length])
    lenXpl = tf.placeholder(tf.int32, [None])

    W1 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), name="W1")
    b1 = tf.Variable(tf.random_normal([n_hidden2]), name="b1")

    W2 = tf.Variable(tf.random_normal([n_hidden2, 1]), name="W2")
    b2 = tf.Variable(tf.random_normal([]), name="b2")

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # dropout is unnecessary.
    # cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=length(X))

    # cost calculating using the tensors.
    encoder_cost_list = []
    encoder_result = []

    #  이걸 gpu 로 빠르게 학습 시키려면, mini_batch를 병렬적으로 연산을 해야할 것이다. <Numpy> 로 계산..
    #  그러나 지금은 일단 이렇게 for loop을 도는 것으로 하자.
    for b_idx in range(n_mini_batch_size):
        #  encoder has 2 layers.
        encoder_layer1_output = tf.matmul(outputs[b_idx], W1) + b1
        encoder_layer1_output = tf.nn.relu(encoder_layer1_output)
        encoder_layer2_output = tf.matmul(encoder_layer1_output, W2) + b2

        #  calculating encoder cost.
        z = np.zeros(max_length)  # same as np.zeros_like(max_length)
        len_ = lenXpl[b_idx]
        ones = tf.ones([len_], dtype=tf.float32) / tf.cast(len_, tf.float32)
        zeros = tf.zeros([max_length - len_])
        onezeros = tf.concat([ones, zeros], axis=0)

        meaningful_result = tf.square((encoder_layer2_output - tf.reshape(Y[b_idx], [tf.shape(Y[b_idx])[0], 1])))
        # L2 Norm 으로 변경 ...
        meaningful_result = tf.multiply(onezeros, meaningful_result)
        encoder_cost_list.append(tf.reduce_sum(meaningful_result))
        encoder_result.append(encoder_layer2_output)  # last time_step encoder prediction result to print

    encoder_result_tf = tf.stack(encoder_result)
    cost_ = tf.stack(encoder_cost_list)
    cost_result = tf.reduce_mean(cost_)

    tf.summary.scalar('/loss', cost_result)
    tf.summary.tensor_summary('/W', W1)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_result)
    merged = tf.summary.merge_all()
    # run the session to calculate the result.

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

train_writer = tf.summary.FileWriter('result/hstrain', sess.graph)
test_writer = tf.summary.FileWriter('result/hstest', sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if restore_session_flag:
    try:
        print("successfully restored the previous session!")
        saver.restore(sess, model_path)
    except:
        print("There is no stored model. I'm creating one at ->", model_path)


if Training_flag:

    h5f = h5py.File('batch_data_h5py/201611_np', 'r')
    batch_size_ = len(h5f['encoder_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)
    encoder_input_batch = h5f['encoder_input_batch'][:]
    encoder_length_vec = h5f['encoder_length_vector_batch'][:]
    encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:], dtype=np.int)
    # encoder_target_batch = h5f['encoder_target_batch'][:]
    encoder_target_batch = (h5f['encoder_target_batch'][:] - 223586273) / 122304694
    h5f.close()

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

    for epoch in range(n_epoch):
        if epoch % 100 == 0 : learning_rate = learning_rate * 0.1
        np.random.shuffle(shuffled_batch_idx)

        lenX = encoder_length_vec[shuffled_batch_idx][:n_mini_batch_size]
        np_lenX = np.asarray(lenX) #.reshape(n_mini_batch_size,1)
        _, loss, summary = sess.run([optimizer, cost_result, merged],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,lenXpl: np_lenX
                             }
                           )
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        train_writer.add_summary(summary, epoch)


    print('Optimization Complete!')
    s = saving_path = saver.save(sess, model_path)
    print('model saving completed!')
    # decoder_input_batch = ( h5f['decoder_input_batch'][:] - 223586273 )/ 122304694
    # pseudo_decoder_target_batch = h5f['decoder_input_batch'][:]  # This is used for initialzation.
    # decoder_target_batch = h5f['decoder_target_batch'][:]


if Test_flag:
    h5f = h5py.File('batch_data_h5py/201610_np', 'r')
    batch_size_ = len(h5f['encoder_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)
    encoder_input_batch = h5f['encoder_input_batch'][:]
    encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:],dtype=np.int)
    encoder_target_batch = h5f['encoder_target_batch'][:]
    h5f.close()
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


    np.random.shuffle(shuffled_batch_idx)
    lenX = encoder_length_vec[shuffled_batch_idx][:n_mini_batch_size]
    encoder_prediction_ , summary = sess.run([encoder_result_tf,merged],
                                                        feed_dict={X: encoder_input_batch[shuffled_batch_idx][
                                                                      :n_mini_batch_size]
                                                            , Y: encoder_target_batch[shuffled_batch_idx][
                                                                      :n_mini_batch_size]
                                                            , lenXpl: lenX
                                                        }
                                  )
    encoder_prediction_ = encoder_prediction_ * 122304694 + 223586273
    encoder_result = []
    re = encoder_prediction_[:, :, 0]
    for i in range(n_mini_batch_size):
        tmp = [int(re[i][lenX[i] - 1]),((encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size][i][lenX[i] - 1]) * 122304694 + 223586273)]
        encoder_result.append(tmp)
    result_np = np.asarray(encoder_result, dtype=np.float32)

    print(result_np)
    cost_of_test_result = np.mean(abs(result_np[:, 0] - result_np[:, 1]))
    print("encoder total cost: ", cost_of_test_result)


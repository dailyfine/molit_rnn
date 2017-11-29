import tensorflow as tf
import numpy as np
import h5py
import os
model_path = "./fcn_tmp/model_test_"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_length = 50
feature_number = 7

which_epoch_to_restore = 9900
restore_session_flag = True
Training_flag = True
Test_flag = True
learning_rate = 0.0001
n_hidden = 10
n_hidden1 = 50
n_hidden2 = 50
n_hidden3 = 1

n_epoch = 1000
n_input = feature_number
n_mini_batch_size = 20
n_decoder_input_feature = 10

# length returns lenX such that
# lenX[idx_specifies_batch] = the true length of times_steps
# i.e., maxlength - zero vectors which is filled in a make_batch method.
# This length info is very useful when calculating the cost removing the effect of zero vector's output.

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

with tf.device('/device:cpu:0'):
    # Declare the model variable and place holder.
    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None])

    W1 = tf.Variable(tf.random_normal([n_input, n_hidden1]), name="W1")
    b1 = tf.Variable(tf.random_normal([n_hidden1]), name="b1")

    W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name="W2")
    b2 = tf.Variable(tf.random_normal([n_hidden2]), name="b2")

    W3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]), name="W2")
    b3 = tf.Variable(tf.random_normal([n_hidden3]), name="b2")

    #  encoder has 2 layers.
    encoder_layer1_output = tf.matmul(X, W1) + b1
    encoder_layer1_output = tf.nn.relu(encoder_layer1_output)
    encoder_layer2_output = tf.matmul(encoder_layer1_output, W2) + b2
    encoder_layer2_output = tf.nn.relu(encoder_layer2_output)

    encoder_layer3_output = tf.matmul(encoder_layer2_output, W3) + b3
    # encoder_layer3_output = tf.nn.relu(encoder_layer3_output)
    encoder_result_tf = encoder_layer3_output

    cost_vec = Y - encoder_layer3_output
    cost = tf.reduce_mean(tf.square(cost_vec))

    training_loss = tf.summary.scalar('/training_loss', cost)
    validation_loss = tf.summary.scalar('/validation_loss', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # run the session to calculate the result.

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

train_writer = tf.summary.FileWriter('result/fcn_train', sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if restore_session_flag:
    try:
        print("successfully restored the previous session!", model_path + str(which_epoch_to_restore))
    except:
        print("There is no stored model. I'm creating one at ->", model_path)

training_data = timecode_generator('201501', '201512')
validation_data = timecode_generator('201601', '201603')
test_data = timecode_generator('201601', '201603')

if Training_flag:
    # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.

    for epoch in range(n_epoch):
        # random 하게, 시간을 고른다.

        if epoch % 100 == 0:
            s = saving_path = saver.save(sess, model_path+"_"+str(epoch))
            print('At epoch #'+str(epoch)+', model saving completed!')

        tr_randint = np.random.randint(0, len(training_data))
        h5f = h5py.File('fcn_batch_data_h5py/'+training_data[tr_randint]+'_np', 'r')

        batch_size_ = len(h5f['fcn_input_batch'][:])
        shuffled_batch_idx = np.arange(batch_size_)
        encoder_input_batch = h5f['fcn_input_batch'][:]
        encoder_target_batch = h5f['fcn_target_batch'][:]
        h5f.close()

        np.random.shuffle(shuffled_batch_idx)
        _, loss, tr_summary = sess.run([optimizer, cost, training_loss],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                                     ,Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                             }
                           )
        print('Epoch: ', '%04d' % (epoch + 1), 'training loss =', '{:.6f}'.format(loss))
        train_writer.add_summary(tr_summary, epoch)

        # Validation ======================================================================
        val_randint = np.random.randint(0, len(validation_data))
        h5f = h5py.File('fcn_batch_data_h5py/'+training_data[tr_randint]+'_np', 'r')

        batch_size_ = len(h5f['fcn_input_batch'][:])
        shuffled_batch_idx = np.arange(batch_size_)
        encoder_input_batch = h5f['fcn_input_batch'][:]
        encoder_target_batch = h5f['fcn_target_batch'][:]
        h5f.close()

        np.random.shuffle(shuffled_batch_idx)
        loss, val_summary = sess.run([cost, validation_loss],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                                     ,Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                             }
                           )
        # print('Epoch:', '%04d' % (epoch + 1), 'validation loss =', '{:.6f}'.format(loss))
        train_writer.add_summary(val_summary, epoch)

    print('Optimization Complete!')



if Test_flag:

    test_randint = np.random.randint(0, len(test_data))
    h5f = h5py.File('fcn_batch_data_h5py/' + training_data[test_randint] + '_np', 'r')

    batch_size_ = len(h5f['fcn_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)
    encoder_input_batch = h5f['fcn_input_batch'][:]
    encoder_target_batch = h5f['fcn_target_batch'][:]
    h5f.close()

    np.random.shuffle(shuffled_batch_idx)
    encoder_prediction_, summary = sess.run([encoder_result_tf, validation_loss],
                                 feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                                     , Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                                     }
                                 )
    # print('Epoch:', '%04d' % (epoch + 1), 'validation loss =', '{:.6f}'.format(loss))

    encoder_prediction_ = encoder_prediction_ * 122304694 + 223586273
    encoder_result = []

    for i in range(n_mini_batch_size):
        tmp = [encoder_prediction_[i],(encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size][i] * 122304694 + 223586273)]
        encoder_result.append(tmp)
    result_np = np.asarray(encoder_result, dtype=np.float32)

    print(result_np)
    cost_of_test_result = np.mean(abs(result_np[:, 0] - result_np[:, 1]))
    print("encoder total cost: ", cost_of_test_result)

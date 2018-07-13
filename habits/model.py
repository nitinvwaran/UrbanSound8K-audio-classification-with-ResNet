import tensorflow as tf
import os, shutil
import numpy as np
from resnet.resnet_model import _building_block_v1, _building_block_v2, _bottleneck_block_v1, _bottleneck_block_v2, \
    batch_norm, \
    conv2d_fixed_padding, block_layer
slim = tf.contrib.slim


class ModelHelper():

    def get_checkpoint_file(self, checkpoint_dir):
        chkpoint_file_path = ''
        with open(checkpoint_dir + 'checkpoint', 'r') as fchk:
            for line in fchk.readlines():
                chkpoint_file_path = line.split(':')[1].strip().replace('"', '')
                break

        return chkpoint_file_path

class AudioEventDetectionResnet(object):

    def build_loss_optimizer(self, logits, num_labels):

        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            # Define loss and optimizer
            ground_truth_input = tf.placeholder(
                tf.int64, [None], name='groundtruth_input')

            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=logits)
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')

            train_step = tf.train.AdamOptimizer(
                learning_rate_input).minimize(cross_entropy_mean)

            predicted_indices = tf.argmax(logits, 1, name="predicted_indices")
            correct_prediction = tf.equal(predicted_indices, ground_truth_input, name='correct_prediction')
            confusion_matrix = tf.confusion_matrix(
                ground_truth_input, predicted_indices, num_classes= num_labels, name = "confusion_matrix")
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="eval_step")

        return ground_truth_input, learning_rate_input, train_step, confusion_matrix, evaluation_step, cross_entropy_mean


    def base_train(self, train_folder, validate_folder, n_train, n_valid, learning_rate, dropoutprob, ncep, nfft, label_count, isTraining, batch_size,
                   epochs, chkpoint_dir, use_nfft, cutoff_spectogram, cutoff_mfcc,bottleneck, num_filters, kernel_size,conv_stride,
                   first_pool_stride,first_pool_size,block_sizes,final_size,resnet_version,data_format):

        with tf.Graph().as_default() as grap:
            logits, fingerprint_input, dropout_prob \
                = self.build_graph(ncep=ncep,nfft=nfft,cutoff_spectogram=cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,use_nfft=use_nfft,is_training=isTraining
                                   ,bottleneck=bottleneck,num_classes = label_count,num_filters = num_filters,kernel_size = kernel_size,conv_stride = conv_stride
                                   ,first_pool_stride = first_pool_stride,first_pool_size = first_pool_size,block_sizes = block_sizes,final_size = final_size,
                                   resnet_version=resnet_version,data_format=data_format)

            ground_truth_input, learning_rate_input, train_step, confusion_matrix, evaluation_step, cross_entropy_mean \
                = self.build_loss_optimizer(logits,label_count)

        with tf.Session(graph=grap) as sess:
            # For checkpoints
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            train_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/train_tensorboard/'
            valid_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/valid_tensorboard/'

            if (os.path.exists(train_tensorboard_dir)):
                shutil.rmtree(train_tensorboard_dir)
            os.mkdir(train_tensorboard_dir)

            if (os.path.exists(valid_tensorboard_dir)):
                shutil.rmtree(valid_tensorboard_dir)
            os.mkdir(valid_tensorboard_dir)

            train_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_tensorboard_dir)


            for i in range(1, epochs + 1):
                total_conf_matrix = None

                xent_summary = 0

                print('Epoch is: ' + str(i))
                j = batch_size
                '''''''''''''''''''''''''''''
                   'Full batch gradient descent'
                   '''''''''''''''''''''''''''''
                while (j <= n_train):
                    npInputs = np.load(
                        train_folder + 'models_label_count_' + str(label_count) + '_numpy_batch' + '_' + str(
                            j) + '.npy')
                    npLabels = np.load(
                        train_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_labels' + '_' + str(
                            j) + '.npy'),

                    print ('Shapes of inputs and labels')
                    print (npInputs.shape)
                    print (npLabels[0].shape)
                    #print (npLabels[0][:10])



                    _, xent_mean, conf_matrix = sess.run(
                        [
                            train_step, cross_entropy_mean, confusion_matrix,

                        ],
                        feed_dict={
                            fingerprint_input: npInputs,
                            ground_truth_input: npLabels[0],
                            learning_rate_input: learning_rate,
                            dropout_prob: dropoutprob,
                        })

                    xent_summary += np.sum(xent_mean)

                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                    xent_train_summary = tf.Summary(
                        value=[tf.Summary.Value(tag="cross_entropy_sum", simple_value=np.sum(xent_mean))])
                    train_writer.add_summary(xent_train_summary,j)

                    if (j == n_train):
                        break

                    if (j + batch_size > n_train):
                        j = n_train
                    else:
                        j = j + batch_size

                'Training set reporting after every epoch'
                print('Training Confusion Matrix:' + '\n' + str(total_conf_matrix))
                true_pos = np.sum(np.diag(total_conf_matrix))
                all_pos = np.sum(total_conf_matrix)
                print('Training Accuracy is: ' + str(float(true_pos / all_pos)))

                loss_train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss_train_summary", simple_value=xent_summary / (n_train))])
                acc_train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_train_summary", simple_value=float(true_pos / all_pos))])

                train_writer.add_summary(loss_train_summary, i)
                train_writer.add_summary(acc_train_summary, i)

                # Save after every 10 epochs
                if (i % 10 == 0):
                    print('Saving checkpoint for epoch:' + str(i))
                    saver.save(sess=sess, save_path=chkpoint_dir + 'base_model_labels_' + str(label_count) + '.ckpt',
                               global_step=i)

                v = batch_size
                valid_conf_matrix = None
                if (batch_size > n_valid):
                    v = n_valid

                while (v <= n_valid):

                    npValInputs = np.load(
                        validate_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_' + str(v) + '.npy')
                    #npValInputs = np.reshape(npValInputs, [-1, max_len * input_size])

                    npValLabels = np.load(
                        validate_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_labels_' + str(
                            v) + '.npy')

                    print (npValInputs.shape)
                    print (npValLabels.shape)
                    print (npValLabels[:,10])

                    _, conf_matrix = sess.run(
                        [evaluation_step, confusion_matrix],
                        feed_dict={
                            fingerprint_input: npValInputs,
                            ground_truth_input: npValLabels,
                            dropout_prob: 1.0
                        })

                    if (valid_conf_matrix is None):
                        valid_conf_matrix = conf_matrix
                    else:
                        valid_conf_matrix += conf_matrix

                    if (v == n_valid):
                        break

                    if (v + batch_size > n_valid):
                        v = n_valid
                    else:
                        v = v + batch_size

                'Validation Set reporting after every epoch'
                print('Validation Confusion Matrix: ' + '\n' + str(valid_conf_matrix))
                true_pos = np.sum(np.diag(valid_conf_matrix))
                all_pos = np.sum(valid_conf_matrix)
                print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))

                acc_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_valid_summary", simple_value=float(true_pos / all_pos))])
                valid_writer.add_summary(acc_valid_summary, i)

    def single_inference(self, nparr, ncep, nfft, cutoff_mfcc,cutoff_spectogram,num_labels,is_training,checkpoint_file_path,use_nfft):


        with tf.Graph().as_default() as grap:
            logits, fingerprint_input, dropout_prob = self.build_graph()

        with tf.Session(graph=grap) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            print('Loading checkpoint file:' + checkpoint_file_path)
            saver = tf.train.import_meta_graph(checkpoint_file_path + '.meta', clear_devices=True)
            saver.restore(sess, checkpoint_file_path)

            # uncomment below to debug variable names
            # between the graph in memory and the graph from checkpoint file
            # get_variable method should now reuse variables from memory scope
            # Variable method was creating new copies and suffixing the numbers to new variables in memory

            # var_name = [v.name for v in tf.global_variables()]

            # print(var_name)
            # reader = pyten.NewCheckpointReader(chk_path)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # print(var_to_shape_map)
            # print(chk_path)

            # saver.restore(sess,chk_path)
            predictions = sess.run(
                [
                    logits
                ],
                feed_dict={
                    fingerprint_input: nparr,
                    dropout_prob: 1.0,
                })

            print('Predictions are:' + str(predictions))
            print('Softmax predictions are:' + str(tf.nn.softmax(predictions)))
            return predictions, tf.nn.softmax(predictions)





    def build_graph(self, ncep, nfft, cutoff_spectogram, cutoff_mfcc, use_nfft,  is_training, bottleneck, num_classes, num_filters,
                    kernel_size, conv_stride, first_pool_stride, first_pool_size, block_sizes, final_size, resnet_version, data_format):


        '''
        print ('Resnet parameters:')
        print ('Resnet Version:' + str(resnet_version))
        print ('Use Nfft:' + str(use_nfft))
        print ('Is Training:' + str(is_training))
        print('Bottleneck:' + str(bottleneck))
        print('Num Classes:' + str(num_classes))
        print('Num Filters:' + str(num_filters))
        print('Kernel Size:' + str(kernel_size))
        print('Conv Stride:' + str(conv_stride))
        print('First Pool Stride:' + str(first_pool_stride))
        print('First Pool Size:' + str(first_pool_size))
        print('Block Sizes:' + str(block_sizes))
        print('Final Size:' + str(final_size))
        print('Resnet Version:' + str(resnet_version))
        print('Data Format:' + str(data_format))
        '''

        if (resnet_version == 1 ):
            block_fn = _building_block_v1
        else:
            block_fn = _building_block_v2

        if (use_nfft):
            input_time_size = cutoff_spectogram
            input_frequency_size = (nfft / 2) + 1
        else:
            input_time_size = cutoff_mfcc
            input_frequency_size = ncep


        print ('Input Time Size:' + str(input_time_size))
        print ('Input Frequency Size:' + str(input_frequency_size))

        # Only declared as expected by the base class
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        with tf.variable_scope('resnet_graph_layer', reuse=tf.AUTO_REUSE):

            fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, input_time_size, input_frequency_size],
                                               name="fingerprint_input")

            # Added 4th dimension for number of channels
            inputs = tf.expand_dims(fingerprint_input,3)


            inputs = conv2d_fixed_padding(inputs = inputs, filters = num_filters,kernel_size = kernel_size,strides = conv_stride,
                                          data_format = data_format)
            inputs = tf.identity(inputs,name='resnet_initial_conv')

            if resnet_version == 1: # resnet version 1
                inputs = batch_norm(inputs = inputs,training = is_training,data_format = data_format)
                inputs = tf.nn.relu(inputs)

            if first_pool_size:
                inputs = tf.layers.max_pooling2d(inputs = inputs, pool_size = first_pool_size, strides = first_pool_stride,
                                                  padding = 'SAME', data_format = data_format)

            for i, num_blocks in enumerate(block_sizes):
                num_filters = num_filters * (2**i)
                inputs = block_layer(inputs = inputs,filters = num_filters, bottleneck= bottleneck,blocks=num_blocks,strides=conv_stride,
                                     block_fn =  block_fn,training= is_training,name='block_layer_{}'.format(i + 1),data_format =  data_format)

            if resnet_version == 2:
                inputs = batch_norm(inputs=inputs,training = is_training,data_format = data_format)
                inputs = tf.nn.relu(inputs)


            axes = [2,3] if data_format == 'channels_first' else [1,2]
            inputs = tf.reduce_mean(inputs, axes,keepdims=True)
            inputs = tf.identity(inputs,name='final_reduce_mean')

            inputs = tf.reshape(inputs, [-1, final_size])
            inputs = tf.layers.dense(inputs=inputs, units= num_classes)
            logits = tf.identity(inputs, 'final_dense')

            #softmax = tf.nn.softmax(logits, name="softmax_op")

            return logits, fingerprint_input, dropout_prob






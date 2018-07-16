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

            loss = tf.reduce_mean(cross_entropy_mean,name="cross_entropy_loss")

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # For batch normalization ops update
            with tf.control_dependencies(extra_update_ops):
                train_step = tf.train.AdamOptimizer(
                    learning_rate_input).minimize(loss)

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
            logits, fingerprint_input, dropout_prob ,is_training \
                = self.build_graph(use_nfft = use_nfft,cutoff_spectogram=cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,nfft=nfft,
                                   ncep=ncep,num_labels=label_count,data_format=data_format)

            ground_truth_input, learning_rate_input, train_step, confusion_matrix, evaluation_step, cross_entropy_mean \
                = self.build_loss_optimizer(logits,label_count)

        with tf.Session(graph=grap) as sess:
            # For checkpoints
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            train_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/train_tensorboard/'
            valid_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/valid_tensorboard/'


            train_tensorboard_dir = '/home/ubuntu/Desktop/urbansound_data/train_tensorboard/'
            valid_tensorboard_dir = '/home/ubuntu/Desktop/urbansound_data/valid_tensorbaord/'


            if (os.path.exists(train_tensorboard_dir)):
                shutil.rmtree(train_tensorboard_dir)
            os.mkdir(train_tensorboard_dir)

            if (os.path.exists(valid_tensorboard_dir)):
                shutil.rmtree(valid_tensorboard_dir)
            os.mkdir(valid_tensorboard_dir)

            train_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_tensorboard_dir)

            print ('Number Train Examples:' + str(n_train))
            print('Number Valid Examples:' + str(n_valid))
            print ('Batch Size:' + str(batch_size))


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
                            j) + '.npy')

                    #print ('Shapes of inputs and labels')
                    #print (npInputs.shape)
                    #print (npLabels.shape)
                    #print (npLabels[0][:10])

                    _, xent_mean, conf_matrix = sess.run(
                        [
                            train_step, cross_entropy_mean, confusion_matrix,

                        ],
                        feed_dict={
                            fingerprint_input: npInputs,
                            ground_truth_input: npLabels,
                            learning_rate_input: learning_rate,
                            dropout_prob: dropoutprob,
                            is_training: True
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

                    #print (npValInputs.shape)
                    #print (npValLabels.shape)
                    #print (npValLabels[:10])

                    _, conf_matrix = sess.run(
                        [evaluation_step, confusion_matrix],
                        feed_dict={
                            fingerprint_input: npValInputs,
                            ground_truth_input: npValLabels,
                            dropout_prob: 1.0,
                            is_training: False
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



    # Adapted from https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    def conv_function(self,x,filter_size, out_channel, strides, pad='SAME',name='conv',data_format = 'channels_last'):

        in_shape = x.get_shape()
        in_channels = in_shape[1] if data_format == 'channels_first' else in_shape[3]
        std = [1,strides,strides,1] if data_format == 'channels_last' else [1,1,strides,strides]
        with tf.variable_scope(name): # different name for each kernel variable
            kernel = tf.get_variable(name='kernel', shape= [filter_size, filter_size, in_channels, out_channel],
                                     dtype = tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
            conv = tf.nn.conv2d(x,filter=kernel,strides=std,padding=pad,name="conv_tensor")

        return conv



    # Adapted from https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    def residual_block_resampled(self,x,filter_size, out_channel,strides,isTraining,data_format='channels_last', name="unit"):

        channel_idx = -1 if data_format == 'channels_last' else 1
        in_channel = x.get_shape().as_list()[channel_idx]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x,name="shortcut")
                else:
                    shortcut = tf.nn.max_pool(value= x, ksize=[1, strides, strides, 1], strides = [1, strides, strides, 1], padding = 'VALID',name="shortcut")
            else:
                shortcut = self.conv_function(x,1,out_channel,strides,name="shortcut")

            x = self.conv_function(x=x,filter_size=filter_size,out_channel = out_channel,strides=strides,name='conv_1')
            x = tf.layers.batch_normalization(inputs=x,training=isTraining,name="bn_1")
            x = tf.nn.relu(features=x,name="relu_1")

            x = self.conv_function(x=x,filter_size = filter_size,out_channel = out_channel,strides=1,name="conv_2")
            x = tf.layers.batch_normalization(inputs=x,training=isTraining,name="bn_2")

            x = x + shortcut # Adds the shortcut link before relu
            x = tf.nn.relu(features=x,name="relu_2")

        return x


    # Adapted from https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    def residual_block_normal(self,x,isTraining,data_format = 'channels_last',name="unit"):

        filter_size = 3
        strides = 1

        channels_idx = -1 if data_format == 'channels_last' else 1
        num_channel = x.get_shape().as_list()[channels_idx]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            shortcut = x # No tf.identity?
            x = self.conv_function(x=x,filter_size=filter_size,out_channel=num_channel,strides=strides,name="conv_1")
            x = tf.layers.batch_normalization(inputs=x,training=isTraining,name="bn_1")
            x = tf.nn.relu(features=x,name="relu_1")

            x = self.conv_function(x=x, filter_size=filter_size, out_channel=num_channel, strides=strides,
                                   name="conv_2")
            x = tf.layers.batch_normalization(inputs=x, training=isTraining, name="bn_2")

            x = x + shortcut
            x = tf.nn.relu(features=x,name="relu_2")

        return x


    def build_graph(self,use_nfft,cutoff_spectogram,cutoff_mfcc,nfft,ncep,num_labels,data_format='channels_last'):

        if (use_nfft):
            input_time_size = cutoff_spectogram
            input_frequency_size = (nfft / 2)
        else:
            input_time_size = cutoff_mfcc
            input_frequency_size = ncep

        print('Input Time Size:' + str(input_time_size))
        print('Input Frequency Size:' + str(input_frequency_size))

        # Only declared as expected by the base class
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, input_time_size, input_frequency_size],
                                           name="fingerprint_input")
        is_training = tf.placeholder(dtype=tf.bool,name="is_training")

        #TODO: Add 4th dimension to the input
        fingerprint_input_4D = tf.expand_dims(fingerprint_input,axis=3,name="fngerprint_4D")
        print (fingerprint_input_4D)

        print ('Building graph ResNet-18')
        channel_sizes = [64, 64, 128, 256, 512]
        kernels_sizes = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        with tf.variable_scope('conv_layer_1'):
            print ('Building unit conv_layer_1')

            x = self.conv_function(x=fingerprint_input_4D,filter_size=kernels_sizes[0],out_channel = channel_sizes[0],strides=strides[0],name="conv_1",data_format=data_format)
            x = tf.layers.batch_normalization(inputs=x,training=is_training,name="bn_1")
            x = tf.nn.relu(features=x,name="relu_1")

            ksize = 3
            strd = 2

            x = tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, strd, strd, 1], 'SAME')


        with tf.variable_scope('conv_layer_2'):
            x = self.residual_block_normal(x = x,name="resnormal_1",isTraining=is_training,data_format=data_format)
            x = self.residual_block_normal(x = x,name="resnormal_2",isTraining=is_training,data_format=data_format)

        with tf.variable_scope('conv_layer_3'):
            x = self.residual_block_resampled(x = x,filter_size = kernels_sizes[2],out_channel = channel_sizes[2],strides=strides[2],
                                              data_format=data_format,name="ressampled_1",isTraining=is_training)
            x = self.residual_block_normal(x=x,data_format=data_format,name="resnormal_1",isTraining=is_training)

        with tf.variable_scope('conv_layer_4'):
            x = self.residual_block_resampled(x = x,filter_size=kernels_sizes[3],out_channel=channel_sizes[3],strides=strides[3],
                                              data_format=data_format,name="ressampled_1",isTraining=is_training)
            x = self.residual_block_normal(x = x,data_format=data_format,name="resnormal_1",isTraining=is_training)

        with tf.variable_scope('conv_layer_5'):
            x = self.residual_block_resampled(x=x,filter_size=kernels_sizes[4],out_channel=channel_sizes[4],strides=strides[4],
                                              data_format=data_format,name='ressampled_1',isTraining=is_training)
            x = self.residual_block_normal(x = x,data_format=data_format,name="resnormal_1",isTraining=is_training)


        with tf.variable_scope('layer_final'):

            mean_axes = [1,2] if data_format == 'channels_last' else [2,3]

            x = tf.reduce_mean(input_tensor=x,axis=mean_axes,name="reduced_tensor")
            x = tf.layers.dense(inputs=x,units=num_labels,use_bias=True)

        return x, fingerprint_input,dropout_prob, is_training





    '''
    def build_graph(self, ncep, nfft, cutoff_spectogram, cutoff_mfcc, use_nfft,  is_training, bottleneck, num_classes, num_filters,
                    kernel_size, conv_stride, first_pool_stride, first_pool_size, block_sizes, final_size, resnet_version, data_format):


        
        #print ('Resnet parameters:')
        #print ('Resnet Version:' + str(resnet_version))
        ##print ('Is Training:' + str(is_training))
        #print('Bottleneck:' + str(bottleneck))
        #print('Num Classes:' + str(num_classes))
        #print('Num Filters:' + str(num_filters))
        #print('Kernel Size:' + str(kernel_size))
        #print('Conv Stride:' + str(conv_stride))
        #print('First Pool Stride:' + str(first_pool_stride))
        #print('First Pool Size:' + str(first_pool_size))
        #print('Block Sizes:' + str(block_sizes))
        #print('Final Size:' + str(final_size))
        #print('Resnet Version:' + str(resnet_version))
        #print('Data Format:' + str(data_format))
        

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

            print (inputs)


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
            '''






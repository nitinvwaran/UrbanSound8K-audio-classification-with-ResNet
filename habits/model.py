import tensorflow as tf
import os, shutil, glob
import numpy as np

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

        return ground_truth_input, learning_rate_input, train_step, confusion_matrix, evaluation_step, cross_entropy_mean, loss


    def base_train(self, train_folder, validate_folder, n_train, n_valid, learning_rate, ncep, nfft, label_count, batch_size,
                   epochs, chkpoint_dir, use_nfft, cutoff_spectogram, cutoff_mfcc,data_format, train_tensorboard_dir, valid_tensorboard_dir):

        with tf.Graph().as_default() as grap:
            logits, fingerprint_input,is_training \
                = self.build_graph(use_nfft = use_nfft,cutoff_spectogram=cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,nfft=nfft,
                                   ncep=ncep,num_labels=label_count,data_format=data_format)

            ground_truth_input, learning_rate_input, train_step, confusion_matrix, evaluation_step, cross_entropy_mean,loss \
                = self.build_loss_optimizer(logits,label_count)

        with tf.Session(graph=grap) as sess:

            # For checkpoints
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            if (os.path.exists(train_tensorboard_dir)):
                shutil.rmtree(train_tensorboard_dir)
            os.mkdir(train_tensorboard_dir)

            if (os.path.exists(valid_tensorboard_dir)):
                shutil.rmtree(valid_tensorboard_dir)
            os.mkdir(valid_tensorboard_dir)

            train_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_tensorboard_dir)

            print ('Batch Size:' + str(batch_size))

            xent_counter = 0

            for i in range(1, epochs + 1):

                total_conf_matrix = None
                valid_conf_matrix = None
                total_loss = 0

                print('Epoch is: ' + str(i))

                '''''''''''''''''''''''''''''
                10-fold cross validation
                '''''''''''''''''''''''''''''
                for val_fold in (1,11):

                    fold_loss = 0

                    for fold in range(1,11):

                        if (fold == val_fold):
                            continue  #to validate nicely, with a smile

                        train_inputs_dir = train_folder + 'fold' + str(fold) + '/batch/inputs/'
                        train_labels_dir = train_folder + 'fold' + str(fold) + '/batch/labels/'

                        os.chdir(train_inputs_dir)
                        print ('Fold directory is:' + train_inputs_dir)

                        for npy_file in glob.glob('*.npy'):

                            npInputs = np.load(train_inputs_dir + npy_file)
                            npLabels = np.load(train_labels_dir + npy_file)

                            print ('Shapes of inputs and labels')
                            print (npInputs.shape)
                            print (npLabels.shape)
                            print (npLabels[:10])

                            _, l, conf_matrix = sess.run(
                                [
                                    train_step, loss, confusion_matrix,

                                ],
                                feed_dict={
                                    fingerprint_input: npInputs,
                                    ground_truth_input: npLabels,
                                    learning_rate_input: learning_rate,
                                    is_training: True
                                })


                            if total_conf_matrix is None:
                                total_conf_matrix = conf_matrix
                            else:
                                total_conf_matrix += conf_matrix

                            total_loss += l

                            xent_counter += 1

                            loss_fold_train_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="loss_train_fold_summary", simple_value=l)])

                            train_writer.add_summary(loss_fold_train_summary, xent_counter)


                    valid_inputs_dir = train_folder + 'fold' + str(val_fold) + '/batch/inputs/'
                    valid_labels_dir = train_folder + 'fold' + str(val_fold) + '/batch/labels/'

                    os.chdir(valid_inputs_dir)
                    print ('Validation fold is:' + valid_inputs_dir)

                    for npy_file in glob.glob('*.npy'):

                        npValInputs = np.load(valid_inputs_dir + npy_file)
                        npValLabels = np.load(valid_labels_dir + npy_file)

                        print ('Shapes of valid inputs and labels')
                        print (npValInputs.shape)
                        print (npValLabels.shape)
                        print (npValLabels[:10])

                        conf_matrix = sess.run(
                             confusion_matrix,
                            feed_dict={
                                fingerprint_input: npValInputs,
                                ground_truth_input: npValLabels,
                                is_training: False
                            })

                        if (valid_conf_matrix is None):
                            valid_conf_matrix = conf_matrix
                        else:
                            valid_conf_matrix += conf_matrix

                'Outside the 10-fold'
                'Training and validation set reporting after every epoch'
                avg_conf_train_matrix = round(total_conf_matrix / 10)
                print('Average Training Confusion Matrix:' + '\n' + str(avg_conf_train_matrix))
                true_pos = np.sum(np.diag(avg_conf_train_matrix))
                all_pos = np.sum(avg_conf_train_matrix)
                print('Training Accuracy is: ' + str(float(true_pos / all_pos)))

                print ('Average training loss is:' + str(total_loss/10))


                acc_train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_train_summary", simple_value=float(true_pos / all_pos))])
                train_writer.add_summary(acc_train_summary, i)

                loss_train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_train_loss", simple_value=total_loss/10)])
                train_writer.add_summary(loss_train_summary, i)

                'Validation Set reporting after every epoch'
                avg_conf_valid_matrix = round(valid_conf_matrix / 10)
                print('Validation Confusion Matrix: ' + '\n' + str(avg_conf_valid_matrix))
                true_pos = np.sum(np.diag(avg_conf_valid_matrix))
                all_pos = np.sum(avg_conf_valid_matrix)
                print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))

                acc_valid_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_valid_summary", simple_value=float(true_pos / all_pos))])
                valid_writer.add_summary(acc_valid_summary, i)

                # Save after every 10 epochs
                if (i % 10 == 0):
                    print('Saving checkpoint for epoch:' + str(i))
                    saver.save(sess=sess, save_path=chkpoint_dir + 'urbansound8k_with_resnet.ckpt',
                               global_step=i)






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
        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, input_time_size, input_frequency_size],
                                           name="fingerprint_input")
        is_training = tf.placeholder(dtype=tf.bool,name="is_training")
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

        return x, fingerprint_input, is_training






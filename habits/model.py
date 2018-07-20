import tensorflow as tf
import os, shutil
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

            print ('Number Train Examples:' + str(n_train))
            print('Number Valid Examples:' + str(n_valid))
            print ('Batch Size:' + str(batch_size))

            xent_counter = 0

            for i in range(1, epochs + 1):

                total_conf_matrix = None

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

                    xent_train_summary = tf.Summary(
                        value=[tf.Summary.Value(tag="cross_entropy_sum", simple_value=l)])
                    xent_counter += 1
                    train_writer.add_summary(xent_train_summary,xent_counter)

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

                acc_train_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="acc_train_summary", simple_value=float(true_pos / all_pos))])

                train_writer.add_summary(acc_train_summary, i)

                # Save after every 10 epochs
                if (i % 10 == 0):
                    print('Saving checkpoint for epoch:' + str(i))
                    saver.save(sess=sess, save_path=chkpoint_dir + 'urbansound8k_with_resnet.ckpt',
                               global_step=i)

                v = batch_size
                valid_conf_matrix = None
                if (batch_size > n_valid):
                    v = n_valid

                while (v <= n_valid):

                    npValInputs = np.load(
                        validate_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_' + str(v) + '.npy')

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


    def do_inference(self,test_batch_directory,ncep,nfft,cutoff_mfcc,cutoff_spectogram,use_nfft,batch_size,checkpoint_dir,label_count,data_format="channels_last"):

        test_count = 0

        # Read the count from the file
        if (os.path.exists(test_batch_directory)):
            with open(test_batch_directory + 'test_count.txt', 'r') as rf:
                for line in rf.readlines():
                    test_count = int(line)


        os.chdir(test_batch_directory)
        print('The Test Directory is:' + str(test_batch_directory))

        with tf.Graph().as_default() as grap:
            logits, fingerprint_input, is_training = self.build_graph(use_nfft=use_nfft,cutoff_spectogram=cutoff_spectogram,cutoff_mfcc = cutoff_mfcc,nfft = nfft,ncep = ncep,num_labels=label_count,data_format=data_format)

        with tf.Session(graph=grap) as sess:

            checkpoint_file_path = checkpoint_dir + 'urbansound8k_with_resnet.ckpt-60'

            print('Checkpoint File is:' + checkpoint_file_path)
            print('Loading Checkpoint File Path')

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file_path)

            j = batch_size

            with open (test_batch_directory + 'ytest.txt','w') as predfile:

                predfile.write('Actual,Prediction' + '\n')

                while (j <= test_count):

                    print ('The batch is:' + str(j))

                    inputs = np.load(test_batch_directory + 'models_label_count_' + str(label_count) + '_numpy_batch_' + str(j) + '.npy')
                    labels = np.load(test_batch_directory + 'models_label_count_' + str(label_count) + '_numpy_batch_labels_' + str(j) + '.npy')

                    predictions = sess.run(logits,
                                           feed_dict={
                                               fingerprint_input: inputs,
                                               is_training: False
                    })

                    soft = tf.nn.softmax(predictions,name="softmax_preds")

                    pred_indexes = tf.argmax(soft,axis=1).eval(session=sess)

                    print ('Shapes of predictions and labels:' + str(labels.shape) + ' ' + str(len(pred_indexes)))
                    output = np.vstack((labels,pred_indexes))
                    t_out = np.asarray(np.transpose(output))

                    #print ('Sample np array:' + str(t_out[:100]))

                    for x in range(0,t_out.shape[0]):
                        predfile.write(str(t_out[x][0]) + ',' +  str(t_out[x][1]) + '\n')

                    if (j == test_count):
                        break

                    if (j + batch_size >= test_count):
                        j = test_count
                    else:
                        j += batch_size


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






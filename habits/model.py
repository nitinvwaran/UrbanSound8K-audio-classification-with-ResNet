import tensorflow as tf
import numpy as np
import habits.inputs_2 as inp
import os
import shutil
import glob
import uuid
from tensorflow.python.framework import graph_util
from audioset import vggish_slim, vggish_params
from resnet.resnet_model import _building_block_v1,_building_block_v2,_bottleneck_block_v1,_bottleneck_block_v2,batch_norm, \
    conv2d_fixed_padding,block_layer
import abc
slim = tf.contrib.slim


class ModelHelper():

    def get_checkpoint_file(self, checkpoint_dir):
        chkpoint_file_path = ''
        with open(checkpoint_dir + 'checkpoint', 'r') as fchk:
            for line in fchk.readlines():
                chkpoint_file_path = line.split(':')[1].strip().replace('"', '')
                break

        return chkpoint_file_path



class AudioEventSuper(object):

    def __init__(self,conf_object):
        self.conf_object = conf_object

    @abc.abstractmethod
    def build_graph(self):
        'Builds the graph'
        return

    @abc.abstractmethod
    def build_final_layer_graph(self,bottleneck_input):
        'Builds last layer for transfer learning'
        return

class AudioEventDetectionVGG():

    def __init__(self, train_vggish, vggish_chkpt_file):
        self.train_vggish = train_vggish
        self.vggish_chkpt_file = vggish_chkpt_file

    def build_graph(self):
        embeddings = vggish_slim.define_vggish_slim(training=self.train_vggish)
        return embeddings

    def build_final_layer_graph(self, label_count, bottleneck_input):
        num_fc_units = 20
        fc = slim.fully_connected(bottleneck_input, num_fc_units)

        logits = slim.fully_connected(
            fc, label_count, activation_fn=None, scope='logits')

        sig_logits = tf.nn.softmax(logits, name='prediction')

        ground_truth = tf.placeholder(dtype=tf.int64, shape=[None])

        return logits, sig_logits, ground_truth

    def retrain(self, label_count, num_epochs, batch_size, train_batch_dir, valid_batch_dir, n_count, n_valid_count):

        with tf.Graph().as_default(), tf.Session() as sess:
            'What is interesting is that google model allows the loaded weights from checkpoint'
            'to not be trained, by setting the flag to false'
            'This allows us to defined the whole model here, train just the new layers, and ' \
            'save the whole graph in one place, instead of the cut and paste from different graphs approach'

            bottleneck_input = vggish_slim.define_vggish_slim(self.train_vggish)
            logits, sig_logits, ground_truth = self.build_final_layer_graph(label_count, bottleneck_input)

            with tf.name_scope('cross_entropy_loss'):
                labels = tf.placeholder(dtype=tf.float32, shape=[None, label_count], name="labels")

                cross_entropy_mean = tf.losses.softmax_cross_entropy(
                    onehot_labels=labels, logits=logits)
                loss_tensor = tf.reduce_mean(cross_entropy_mean, name='loss_op')

            with tf.name_scope('train'):
                optimizer = tf.train.MomentumOptimizer(
                    vggish_params.LEARNING_RATE, momentum=0.9, use_nesterov=False)
                training_op = optimizer.minimize(loss_tensor, name='train_op')

            '''
            with tf.name_scope('cross_entropy_loss'):

                labels = tf.placeholder(dtype=tf.float32, shape=[None, label_count], name="labels")

                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels, name='xent')
                loss_tensor = tf.reduce_mean(xent, name='loss_op')

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=vggish_params.LEARNING_RATE,
                    epsilon=vggish_params.ADAM_EPSILON)
                training_op = optimizer.minimize(loss_tensor, name='train_op')
            '''

            predicted_indices = tf.argmax(sig_logits, 1)
            correct_prediction = tf.equal(predicted_indices, ground_truth)
            confusion_matrix = tf.confusion_matrix(
                ground_truth, predicted_indices, num_classes=label_count)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess.run(tf.global_variables_initializer())
            vggish_slim.load_vggish_slim_checkpoint(sess, self.vggish_chkpt_file)

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            # labels_tensor = sess.graph.get_tensor_by_name('cross_entropy_loss/labels:0')
            # loss_tensor = sess.graph.get_tensor_by_name('cross_entropy_loss/loss_op:0')
            # training_op = sess.graph.get_operation_by_name('train/train_op')

            dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels))
            dataset = dataset.batch(batch_size=batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels))
            val_dataset = val_dataset.batch(batch_size=batch_size)

            iterator = dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()

            for i in range(1, num_epochs + 1):
                total_conf_matrix = None

                # j = batch_size
                # while (j <= n_count):

                npInputs = np.load(train_batch_dir + 'vgg_embedding_batch.npy')
                npLabels = np.load(train_batch_dir + 'vgg_embedding_batch_label_hot.npy')

                # npInputs = np.load(train_batch_dir + 'vgg_embedding_batch' + str(j) + '.npy')
                # npLabels = np.load(train_batch_dir + 'vgg_embedding_batch_label_hot' + str(j) + '.npy')
                # npLabelsTruth = np.load(train_batch_dir + 'vgg_embedding_batch_label' + str(j) + '.npy')

                # print ('Batch is:' + str(j))
                # print (npInputs.shape)
                # print (npLabels.shape)

                sess.run(iterator.initializer, feed_dict={
                    features_tensor: npInputs,
                    labels: npLabels
                })

                train, loss, sigmoid, conf_matrix = sess.run(
                    [
                        training_op, loss_tensor, sig_logits, confusion_matrix
                    ],
                    feed_dict={
                        features_tensor: npInputs,
                        labels: npLabels,
                        # ground_truth:npLabelsTruth

                    }
                )
                # print('Train Labels:' + str(npLabelsTruth))
                # print('Train Labels Hot:' + str(npLabels))
                # print('The train sigmoid is:' + str(sigmoid))

                # if total_conf_matrix is None:
                #    total_conf_matrix = conf_matrix
                # else:
                #    total_conf_matrix += conf_matrix

                # if (j == n_count):
                #    break

                # if ( j + batch_size > n_count):
                #    j = n_count
                # else:
                #    j += batch_size

                print('Epoch:' + str(i))
                print('Training Confusion Matrix:' + '\n' + str(conf_matrix))
                true_pos_train = np.sum(np.diag(conf_matrix))
                all_pos_train = np.sum(conf_matrix)
                print(' Train Accuracy is: ' + str(float(true_pos_train / all_pos_train)))

                # Save after every 10 epochs
                # if (i % 10 == 0):
                # print('Saving checkpoint')
                # saver.save(sess=sess, save_path=retrain_chkpoint_dir + 'model_labels_' + str(label_count) + '.ckpt',
                #           global_step=i)

                # Validation set reporting
                # print ('Validation on batch:' + str(v))

                # v = batch_size

                # if (v > n_valid_count):
                #    v = n_valid_count

                # valid_conf_matrix = None
                # while (v <= n_valid_count):

                npInputsVal = np.load(valid_batch_dir + 'vgg_embedding_batch.npy')
                npLabelsVal = np.load(valid_batch_dir + 'vgg_embedding_batch_label_hot.npy')

                #npInputsVal = np.load(valid_batch_dir + 'vgg_embedding_batch' + str(v) + '.npy')
                #npLabelsVal = np.load(valid_batch_dir + 'vgg_embedding_batch_label_hot' + str(v) + '.npy')
                #npLabelsValTruth = np.load(valid_batch_dir + 'vgg_embedding_batch_label' + str(v) + '.npy')

                #print('Validation batch:' + str(v))
                # print (npInputsVal.shape)
                # print (npLabelsVal.shape)

                # sess.run(val_iterator.initializer,feed_dict={
                #        features_tensor: npInputsVal,
                #        labels: npLabelsVal,
                #    })

                sigmoid_val, val_conf_matrix = sess.run(
                    [sig_logits, confusion_matrix],
                    feed_dict={
                        features_tensor: npInputsVal,
                        labels: npLabelsVal,
                        #ground_truth: npLabelsValTruth
                    }
                )

                # print('Val Labels:' + str(npLabelsValTruth))
                # print('Val Labels Hot:' + str(npLabelsVal))

                # print('The validation sigmoid is:' + str(sigmoid_val))

                # if (valid_conf_matrix is None):
                #    valid_conf_matrix = conf_matrix
                # else:
                #    valid_conf_matrix += conf_matrix

                # if (v == n_valid_count):
                #    break

                # if (v + batch_size > n_valid_count):
                #    v = n_valid_count
                # else:
                #    v += batch_size

                print('Validation Confusion Matrix: ' + '\n' + str(val_conf_matrix))
                true_pos = np.sum(np.diag(val_conf_matrix))
                all_pos = np.sum(val_conf_matrix)
                print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))






class AudioEventDetection(object):

    def __init__(self,resnet_size,bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               final_size, resnet_version=2, data_format=None,
                dtype=tf.float32):

        _BATCH_NORM_DECAY = 0.997
        _BATCH_NORM_EPSILON = 1e-5
        DEFAULT_VERSION = 2
        DEFAULT_DTYPE = tf.float32
        CASTABLE_TYPES = (tf.float16,)
        ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

        """Creates a model for classifying an image.
            Args:
              resnet_size: A single integer for the size of the ResNet model.
              bottleneck: Use regular blocks or bottleneck blocks.
              num_classes: The number of classes used as labels.
              num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent block
                layer.
              kernel_size: The kernel size to use for convolution.
              conv_stride: stride size for the initial convolutional layer
              first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
              first_pool_stride: stride size for the first pooling layer. Not used
                if first_pool_size is None.
              block_sizes: A list containing n values, where n is the number of sets of
                block layers desired. Each value should be the number of blocks in the
                i-th set.
              block_strides: List of integers representing the desired stride size for
                each of the sets of block layers. Should be same length as block_sizes.
              final_size: The expected size of the model after the second pooling.
              resnet_version: Integer representing which version of the ResNet network
                to use. See README for details. Valid values: [1, 2]
              data_format: Input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available.
              dtype: The TensorFlow dtype to use for calculations. If not specified
                tf.float32 is used.
            Raises:
              ValueError: if invalid version is selected.
            """
        self.resnet_size = resnet_size

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def build_final_layer_graph(self, label_count, isTraining, bottleneck_input):

        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Builds the final layer with number of cells = label count, this will need to be retrained with every new label
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        second_fc_output_channels = 128
        ground_truth_name = 'ground_truth_retrain_label_' + str(label_count)

        with tf.variable_scope('layer_four', reuse=tf.AUTO_REUSE):
            l4b_init = tf.random_normal_initializer(mean=0, stddev=0.0, dtype=tf.float32)
            l4w_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
            final_fc_weights = tf.get_variable(name="weight_four",
                                               shape=[second_fc_output_channels, label_count], dtype=tf.float32,
                                               initializer=l4w_init)
            final_fc_bias = tf.get_variable(name="bias_four", shape=[label_count], dtype=tf.float32,
                                            initializer=l4b_init)
            final_fc = tf.matmul(bottleneck_input, final_fc_weights) + final_fc_bias

        ground_truth_retrain_input = tf.placeholder(dtype=tf.int64, shape=[None], name=ground_truth_name)

        # The final result - a softmax can be applied to this for inference
        return final_fc, bottleneck_input, ground_truth_retrain_input, final_fc_weights, final_fc_bias




    def build_graph(self,fingerprint_input, dropout_prob, ncep, nfft, max_len, isTraining, use_nfft=True):

        if (use_nfft):  # nfft == spectogram
            input_frequency_size = nfft
        else:
            input_frequency_size = ncep

        input_time_size = max_len

        with tf.variable_scope('layer_resnet',reuse=tf.AUTO_REUSE):
            fingerprint_4d = tf.reshape(fingerprint_input,
                                        [-1, input_time_size, input_frequency_size, 1])

            # Channels First
            fingerprint_t = tf.transpose(fingerprint_4d,[0,3,1,2])

            inputs = conv2d_fixed_padding(
                inputs=fingerprint_t, filters=self.num_filters, kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)

            if self.resnet_version == 1:
                inputs = batch_norm(inputs, isTraining, self.data_format)
                inputs = tf.nn.relu(inputs)

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=isTraining,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = batch_norm(inputs, isTraining, self.data_format)
                inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.reshape(inputs, [-1, self.final_size])
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)

            inputs = tf.identity(inputs, 'final_dense')
            return inputs


    '''
    def build_graph(self, fingerprint_input, dropout_prob, ncep, nfft, max_len, isTraining, use_nfft=True):

        
        'Contains the core NN Architecture of the class. This can be replaced with any other architecture coded from scratch,' 
        'Or a pre-trained model from which the relevant bottleneck tensors,  are extracted and fed'
        'to the final layer graph function above for transfer learning / retraining'
        'Extracting tensors at a layer that is not the bottleneck, will need changes to final layer graph function'
        'As more tensors need to be fed to the final layer or the layers before it (depending on retraining judgements)'

        if (use_nfft):  # nfft == spectogram
            input_frequency_size = nfft
        else:
            input_frequency_size = ncep

        input_time_size = max_len

        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])

        # Hyper Parameters!
        first_filter_width = 8
        first_filter_height = input_time_size
        first_filter_count = 186
        first_filter_stride_x = 1
        first_filter_stride_y = 1

        with tf.variable_scope('layer_one', reuse=tf.AUTO_REUSE):

            l1b_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
            l1w_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)

            first_weights = tf.get_variable(name="weight_one",
                                            shape=[first_filter_height, first_filter_width, 1, first_filter_count],
                                            dtype=tf.float32,
                                            initializer=l1w_init,
                                            )
            first_bias = tf.get_variable(name="bias_one", shape=[first_filter_count], dtype=tf.float32,
                                         initializer=l1b_init)
            first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                                      [1, first_filter_stride_y, first_filter_stride_x, 1], 'VALID') + first_bias
            first_relu = tf.nn.relu(first_conv)

            if isTraining:
                first_dropout = tf.nn.dropout(first_relu, dropout_prob)
            else:
                first_dropout = first_relu

        first_conv_output_width = math.floor(
            (input_frequency_size - first_filter_width + first_filter_stride_x) /
            first_filter_stride_x)
        first_conv_output_height = math.floor(
            (input_time_size - first_filter_height + first_filter_stride_y) /
            first_filter_stride_y)
        first_conv_element_count = int(
            first_conv_output_width * first_conv_output_height * first_filter_count)
        flattened_first_conv = tf.reshape(first_dropout, [-1, first_conv_element_count])

        first_fc_output_channels = 128

        with tf.variable_scope('layer_two', reuse=tf.AUTO_REUSE):
            l2b_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
            l2w_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)

            first_fc_weights = tf.get_variable(name="weight_two",
                                               shape=[first_conv_element_count, first_fc_output_channels],
                                               dtype=tf.float32, initializer=l2w_init)
            first_fc_bias = tf.get_variable(name="bias_two", shape=[first_fc_output_channels], dtype=tf.float32,
                                            initializer=l2b_init)
            first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias

            if isTraining:
                second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
            else:
                second_fc_input = first_fc

        second_fc_output_channels = 128  # number of cols in bottleneck tensor

        with tf.variable_scope('layer_three', reuse=tf.AUTO_REUSE):
            l3b_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
            l3w_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)

            second_fc_weights = tf.get_variable(name="weight_three",
                                                shape=[first_fc_output_channels, second_fc_output_channels],
                                                dtype=tf.float32, initializer=l3w_init)
            second_fc_bias = tf.get_variable(name="bias_three", shape=[second_fc_output_channels], dtype=tf.float32,
                                             initializer=l3b_init)
            second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias

            if isTraining:
                final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
            else:
                final_fc_input = second_fc

        # The bottleneck tensor is final_fc_input
        return final_fc_input, first_weights, first_bias, first_fc_weights, first_fc_bias, second_fc_weights, second_fc_bias
        
    '''

    def inference_frozen(self, nparr, frozen_graph):

        # TODO: dead function, reuse for when saving frozen graph is automated. This should also get its own common class..

        gph = self.load_graph(frozen_graph)
        y = gph.get_tensor_by_name('prefix/labels_softmax:0')

        fingerprint_input = gph.get_tensor_by_name('prefix/fingerprint_input:0')

        with tf.Session(graph=gph) as sess:
            y_out = sess.run(
                [
                    y
                ],
                feed_dict={
                    fingerprint_input: nparr
                    # dropout_prob: 1.0,
                })

        return y_out

    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def save_frozen_graph(self, chkpoint_dir, max_len, ncep, label_count, isTraining):

        # TODO: dead function, reuse for when saving frozen graph is automated. This should also get its own common class

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        logits = self.build_graph(fingerprint_input=fingerprint_input, dropout_prob=dropout_prob, ncep=ncep,
                                  max_len=max_len,
                                  isTraining=isTraining)

        tf.nn.softmax(logits, name="labels_softmax")

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=chkpoint_dir)
        chk_path = checkpoint.model_checkpoint_path
        print(chk_path)

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()

        saver.restore(sess, chk_path)

        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['labels_softmax'])
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname(chkpoint_dir),
            os.path.basename(chkpoint_dir + 'habits_frozen.pb'),
            as_text=False)

    def inference(self, ncep, nfft, cutoff_mfcc, cutoff_spectogram, label_count, isTraining, nparr,
                  checkpoint_file_path, use_nfft=True):

        if (use_nfft):
            max_len = cutoff_spectogram
            input_size = nfft
        else:
            max_len = cutoff_mfcc
            input_size = ncep

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * input_size],
                                           name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        logits = self.build_graph(fingerprint_input=fingerprint_input,
                                            dropout_prob=dropout_prob, ncep=ncep, nfft=nfft,
                                            max_len=max_len, isTraining=isTraining, use_nfft=use_nfft)
        '''
        bottleneck_input, _, _, _, _, _, _ = self.build_graph(fingerprint_input=fingerprint_input,
                                                              dropout_prob=dropout_prob, ncep=ncep, nfft=nfft,
                                                              max_len=max_len, isTraining=isTraining, use_nfft=use_nfft)
        logits, _, _, _, _ = self.build_final_layer_graph(label_count=label_count, isTraining=isTraining,
                                                       bottleneck_input=bottleneck_input)
        '''

        init = tf.global_variables_initializer()
        sess = tf.Session()
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
        # print ('Softmax predictions are:' + tf.nn.softmax(logits))
        return predictions



    def create_bottlenecks_cache(self, file_dir, bottleneck_input_dir, ncep, nfft, cutoff_mfcc, cutoff_spectogram,
                                 isTraining, base_chkpoint_dir, label_count, labels_meta_file, use_nfft=True):

        model_helper = ModelHelper()

        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        'Creates numpy files on disk for all training data that represents bottleneck transformation (.npy file) for each wav file'
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        if (not os.path.exists(base_chkpoint_dir + 'checkpoint')):
            raise Exception('Base Checkpoint File is Missing! A Crisis!')

        if (use_nfft):
            input_size = nfft
            max_len = cutoff_spectogram
        else:
            input_size = ncep
            max_len = cutoff_mfcc

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * input_size],
                                           name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        bottleneck_input, _, _, _, _, _, _ = self.build_graph(fingerprint_input=fingerprint_input,
                                                              dropout_prob=dropout_prob,
                                                              ncep=ncep, nfft=nfft, max_len=max_len,
                                                              isTraining=isTraining, use_nfft=use_nfft)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        base_chkpoint_file_path = model_helper.get_checkpoint_file(base_chkpoint_dir)
        print('Base Checkpoint File is:' + base_chkpoint_file_path)

        saver = tf.train.import_meta_graph(base_chkpoint_file_path + '.meta', clear_devices=True)
        saver.restore(sess, base_chkpoint_file_path)

        ''''
        # uncomment below to debug variable names
        # between the graph in memory and the graph from checkpoint file
        # get_variable method should now reuse variables from memory scope
        # Variable method was creating new copies and suffixing the numbers to new variables in memory

        var_name = [v.name for v in tf.global_variables()]
        print(var_name)
        reader = pyten.NewCheckpointReader(chk_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        print(var_to_shape_map)
        print(chk_path)
        saver.restore(sess,chk_path)
        '''

        _, labels_meta = inp.get_labels_and_count(labels_meta_file)

        # Reset bottleneck input directory
        if (os.path.exists(bottleneck_input_dir)):
            shutil.rmtree(bottleneck_input_dir)

        os.makedirs(bottleneck_input_dir)

        os.chdir(file_dir)
        file_count = 0
        for filename in glob.glob('*.wav'):

            file_count = file_count + 1
            mfcc, spec = inp.prepare_mfcc_spectogram(file_dir=file_dir, file_name=filename, ncep=ncep, nfft=nfft,
                                                     cutoff_mfcc=cutoff_mfcc, cutoff_spectogram=cutoff_spectogram)

            if (use_nfft):
                nparr2 = np.reshape(spec, [-1, cutoff_spectogram * input_size])
            else:
                nparr2 = np.reshape(mfcc, [-1, cutoff_mfcc * input_size])

            bottleneck = sess.run(
                [
                    bottleneck_input
                ],
                feed_dict={
                    fingerprint_input: nparr2,
                    dropout_prob: 1.0,
                })

            np.save(bottleneck_input_dir + 'numpy_bottle_' + filename + '.npy', bottleneck[0])

            labels = []
            # l = inp.stamp_label(num_labels=label_count,labels_meta=labels_meta,filename=filename)
            labels.append(inp.stamp_label(num_labels=label_count, labels_meta=labels_meta, filename=filename))
            np.save(bottleneck_input_dir + 'numpy_bottle_labels_' + filename + '.npy', np.array(labels))

        return file_count

    def rebuild_graph_post_transfer_learn(self, ncep, nfft, cutoff_mfcc, cutoff_spectogram, label_count, isTraining,
                                          chkpoint_dir, base_chkpoint_file, version_chkpoint_file, use_nfft=True):

        if (use_nfft):
            input_size = nfft
            max_len = cutoff_spectogram
        else:
            input_size = ncep
            max_len = cutoff_mfcc

        with tf.Graph().as_default() as grap:

            fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * input_size],
                                               name="fingerprint_input")
            dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            bottleneck_tensor, first_weights, first_bias, first_fc_weights, first_fc_bias, \
            second_fc_weights, second_fc_bias = self.build_graph(fingerprint_input=fingerprint_input,
                                                                 dropout_prob=dropout_prob,
                                                                 ncep=ncep, nfft=nfft, max_len=max_len,
                                                                 isTraining=isTraining, use_nfft=use_nfft)

            final_fc, _, _, _, _ = self.build_final_layer_graph(label_count=label_count, isTraining=isTraining,
                                                                bottleneck_input=bottleneck_tensor)

            varb = [t for t in grap.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                    not t.name.__contains__('layer_four')]
            varb2 = [t for t in grap.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                     t.name.__contains__('layer_four')]
            varball = [t for t in grap.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        with tf.Session(graph=grap) as sess:

            saver = tf.train.Saver(varb2)
            saver.restore(sess, version_chkpoint_file)
            saver = tf.train.Saver(varb)
            saver.restore(sess, base_chkpoint_file)

            'Useful debugging'
            'val = [sess.run(v) for v in tf.global_variables()]'

            saverFinal = tf.train.Saver(
                varball)  # Turns out this new saver is the ley to combining graphs into a new graph / checkpoint
            print('Saving checkpoint')
            saverFinal.save(sess=sess, save_path=chkpoint_dir + 'habits/' + 'transfer_model_label_count_' + str(
                label_count) + '.ckpt')
            print(
                'Saved new graph version to location:' + chkpoint_dir + 'habits/' + 'transfer_model_label_count_' + str(
                    label_count) + '.ckpt')

            'Use the below to check whether the new graph is being stitched together correctly'
            # saverFinal.save(sess=sess, save_path='/home/nitin/PycharmProjects/habits/checkpoints/test.ckpt')

        # Use the below block to check whether the new graph was stitched together correctly
        '''''
        with tf.Session(graph=grap) as sess2:

            init = tf.global_variables_initializer()
            sess2.run(init)

            print ('Verifying the correct graph')
            saver = tf.train.import_meta_graph('/home/nitin/PycharmProjects/habits/checkpoints/test.ckpt' + '.meta', clear_devices=True)
        saver.restore(sess2, '/home/nitin/PycharmProjects/habits/checkpoints/test.ckpt')


            val = [sess2.run(v) for v in tf.global_variables()]
        '''''

    def retrain(self, train_bottleneck_dir, valid_bottleneck_dir, label_count, ncep, nfft, cutoff_mfcc,
                cutoff_spectogram,
                isTraining, batch_size, n_count, n_valid_count, epochs, chkpoint_dir, learning_input=0.001,
                use_nfft=True):

        print(
            'Starting re-training with following parameters:' + ' train count: ' + str(n_count) + ' valid count ' + str(
                n_valid_count))

        uu_id = uuid.uuid4()
        retrain_chkpoint_dir = chkpoint_dir + 'tmp/version_model_labels_' + str(label_count) + '_time_' + str(
            uu_id) + '/'

        with tf.Graph().as_default() as grap:

            second_fc_output_channels = 128

            bottleneck_input = tf.placeholder(dtype=tf.float32, shape=[None, second_fc_output_channels],
                                              name='bottleneck_input')
            final_fc, bottleneck_input, ground_truth_retrain_input, final_fc_weights, final_fc_bias \
                = self.build_final_layer_graph(label_count=label_count, isTraining=isTraining,
                                               bottleneck_input=bottleneck_input)

            with tf.name_scope('cross_entropy'):
                cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                    labels=ground_truth_retrain_input, logits=final_fc)

            with tf.name_scope('train'):
                learning_rate_input = tf.placeholder(
                    tf.float32, [], name='learning_rate_input')
                train_step = tf.train.GradientDescentOptimizer(
                    learning_rate_input).minimize(cross_entropy_mean)

            predicted_indices = tf.argmax(final_fc, 1)
            correct_prediction = tf.equal(predicted_indices, ground_truth_retrain_input)
            confusion_matrix = tf.confusion_matrix(
                ground_truth_retrain_input, predicted_indices, num_classes=label_count)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session(graph=grap) as sess:

            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            '''
            ' Useful debugging below ' 
            # Check that only part of the graph is restored 
            var_name = [v.name for v in tf.global_variables()]
            print(var_name)

            varb = grap.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print (varb)

            val = [sess.run(v) for v in varb]
            print(val)
            '''

            for i in range(1, epochs + 1):
                total_conf_matrix = None

                j = batch_size
                while (j <= n_count):

                    # print ('Training batch:' + str(j))

                    npInputs = np.load(train_bottleneck_dir + 'bottleneck_batch' + '_' + str(j) + '.npy')
                    npLabels = np.load(train_bottleneck_dir + 'bottleneck_batch_label' + '_' + str(j) + '.npy')

                    npInputs2 = np.reshape(npInputs, [npInputs.shape[0] * npInputs.shape[1], npInputs.shape[2]])
                    npLabels2 = np.reshape(npLabels, [npLabels.shape[0]])

                    xent_mean, _, conf_matrix = sess.run(
                        [
                            cross_entropy_mean, train_step,
                            confusion_matrix
                        ],
                        feed_dict={
                            bottleneck_input: npInputs2,
                            ground_truth_retrain_input: npLabels2,
                            learning_rate_input: learning_input,
                        })

                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                    if (j == n_count):
                        break

                    if (j + batch_size > n_count):
                        j = n_count
                    else:
                        j = j + batch_size

                print('Epoch:' + str(i))
                print('Training Confusion Matrix:' + '\n' + str(total_conf_matrix))

                # Save after every 10 epochs
                if (i % 10 == 0):
                    print('Saving checkpoint')
                    saver.save(sess=sess, save_path=retrain_chkpoint_dir + 'model_labels_' + str(label_count) + '.ckpt',
                               global_step=i)

                v = batch_size
                valid_conf_matrix = None

                if (batch_size > n_valid_count):
                    v = n_valid_count

                while (v <= n_valid_count):

                    # Validation set reporting
                    # print ('Validation on batch:' + str(v))

                    npValInputs = np.load(valid_bottleneck_dir + 'bottleneck_batch_' + str(v) + '.npy')
                    npValLabels = np.load(valid_bottleneck_dir + 'bottleneck_batch_label_' + str(v) + '.npy')

                    npValInputs2 = np.reshape(npValInputs,
                                              [npValInputs.shape[0] * npValInputs.shape[1], npValInputs.shape[2]])
                    npValLabels2 = np.reshape(npValLabels, [npValLabels.shape[0]])

                    _, conf_matrix, _ = sess.run(
                        [evaluation_step, confusion_matrix, predicted_indices],
                        feed_dict={
                            bottleneck_input: npValInputs2,
                            ground_truth_retrain_input: npValLabels2,

                        })

                    if (valid_conf_matrix is None):
                        valid_conf_matrix = conf_matrix
                    else:
                        valid_conf_matrix += conf_matrix

                    if (v == n_valid_count):
                        break

                    if (v + batch_size > n_valid_count):
                        v = n_valid_count
                    else:
                        v = v + batch_size

                print('Validation Confusion Matrix: ' + '\n' + str(valid_conf_matrix))
                true_pos = np.sum(np.diag(valid_conf_matrix))
                all_pos = np.sum(valid_conf_matrix)
                print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))

        print('Creating new complete graph')
        base_checkpoint_file = model_helper.get_checkpoint_file(checkpoint_dir=chkpoint_dir + 'base_dir/')
        version_checkpoint_file = model_helper.get_checkpoint_file(checkpoint_dir=retrain_chkpoint_dir)

        print('Base checkpoint graph is:' + base_checkpoint_file)
        print('Version checkpoint graph is:' + version_checkpoint_file)

        # Should go back to the habits.py module
        self.rebuild_graph_post_transfer_learn(ncep=ncep, nfft=nfft, cutoff_mfcc=cutoff_mfcc,
                                               cutoff_spectogram=cutoff_spectogram, label_count=label_count,
                                               isTraining=isTraining
                                               , base_chkpoint_file=base_checkpoint_file,
                                               version_chkpoint_file=version_checkpoint_file, use_nfft=use_nfft,
                                               chkpoint_dir=chkpoint_dir)

    def base_train(self, ncep, nfft, max_len, label_count, isTraining, batch_size, train_folder, validate_folder,
                   n_train, n_valid, epochs, chkpoint_dir, use_nfft=True, learning_rate=0.001, dropoutprob=0.5):

        sess = tf.InteractiveSession()

        if (use_nfft):
            input_size = nfft
        else:
            input_size = ncep

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * input_size],
                                           name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        bottleneck_input, _, _, _, _, _, _ = self.build_graph(fingerprint_input=fingerprint_input,
                                                              dropout_prob=dropout_prob, ncep=ncep, nfft=nfft,
                                                              max_len=max_len, isTraining=isTraining, use_nfft=use_nfft)
        final_fc, _, _, _, _ = self.build_final_layer_graph(label_count=label_count, isTraining=isTraining,
                                                            bottleneck_input=bottleneck_input)

        # Define loss and optimizer
        ground_truth_input = tf.placeholder(
            tf.int64, [None], name='groundtruth_input')

        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=final_fc)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        with tf.name_scope('train'):
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            train_step = tf.train.GradientDescentOptimizer(
                learning_rate_input).minimize(cross_entropy_mean)

        predicted_indices = tf.argmax(final_fc, 1)
        correct_prediction = tf.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.confusion_matrix(
            ground_truth_input, predicted_indices, num_classes=label_count)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)

        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)

        # For checkpoints
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(1, epochs + 1):
            total_conf_matrix = None

            print('Epoch is: ' + str(i))

            j = batch_size
            while (j <= n_train):

                npInputs = np.load(
                    train_folder + 'models_label_count_' + str(label_count) + '_numpy_batch' + '_' + str(j) + '.npy')
                npLabels = np.load(
                    train_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_labels' + '_' + str(
                        j) + '.npy'),

                npInputs2 = np.reshape(npInputs, [-1, max_len * input_size])
                xent_mean, _, _, conf_matrix = sess.run(
                    [
                        cross_entropy_mean, train_step,
                        increment_global_step, confusion_matrix
                    ],
                    feed_dict={
                        fingerprint_input: npInputs2,
                        ground_truth_input: npLabels[0],
                        learning_rate_input: learning_rate,
                        dropout_prob: dropoutprob,
                    })

                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix

                if (j == n_train):
                    break

                if (j + batch_size > n_train):
                    j = n_train
                else:
                    j = j + batch_size

            print('Training Confusion Matrix:' + '\n' + str(total_conf_matrix))

            # Save after every 10 epochs
            if (i % 10 == 0):
                print('Saving checkpoint for epoch:' + str(i))
                saver.save(sess=sess, save_path=chkpoint_dir + 'base_model_labels_' + str(label_count) + '.ckpt',
                           global_step=i)

            # Validation set reporting
            v = batch_size
            valid_conf_matrix = None
            if (batch_size > n_valid):
                v = n_valid

            while (v <= n_valid):

                npValInputs = np.load(
                    validate_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_' + str(v) + '.npy')
                npValInputs = np.reshape(npValInputs, [-1, max_len * input_size])

                npValLabels = np.load(
                    validate_folder + 'models_label_count_' + str(label_count) + '_numpy_batch_labels_' + str(
                        v) + '.npy')
                test_accuracy, conf_matrix, pred_indices = sess.run(
                    [evaluation_step, confusion_matrix, predicted_indices],
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

            print('Validation Confusion Matrix: ' + '\n' + str(valid_conf_matrix))
            true_pos = np.sum(np.diag(valid_conf_matrix))
            all_pos = np.sum(valid_conf_matrix)
            print(' Validation Accuracy is: ' + str(float(true_pos / all_pos)))


if __name__ == '__main__':
    vggish_chkpt_file = '/home/ubuntu/Desktop/aws_habits/FMSG_Habits/audioset/vggish_model.ckpt'
    train_batch_dir = '/home/ubuntu/Desktop/vggish/train_batch/numpy_batch/vgg_embedding_batch/'
    valid_batch_dir = '/home/ubuntu/Desktop/vggish/valid_batch/numpy_batch/vgg_embedding_batch/'
    n_count = 6624
    n_valid_count = 2187

    model_vgg = AudioEventDetectionVGG(train_vggish=False, vggish_chkpt_file=vggish_chkpt_file)
    model_vgg.retrain(label_count=4, num_epochs=20, batch_size=500, train_batch_dir=train_batch_dir,
                      valid_batch_dir=valid_batch_dir, n_count=n_count, n_valid_count=n_valid_count)
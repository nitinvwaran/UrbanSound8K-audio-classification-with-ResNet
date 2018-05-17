
import math
import tensorflow as tf
import numpy as np
import habits.inputs_2 as inp
import argparse
import sys
import os
from tensorflow.python.framework import graph_util


def build_final_layer_graph(label_count,ncep,max_len,isTraining,bottleneck_input):

    '''
    Builds the final layer, this will need to be retrained with every new label
    '''

    print ('bottleneck_input shape:' + str(bottleneck_input.shape))
    second_fc_output_channels = 128
    ground_truth_name = 'ground_truth_retrain_label_' + str(label_count)
    bottleneck_input_name = 'bottleneck_input_label_' + str(label_count)

    name_scope = 'layer_retrain_labels'
    with tf.name_scope(name_scope):
        bottleneck_input = tf.placeholder_with_default(input=bottleneck_input, shape=[None, second_fc_output_channels],name = bottleneck_input_name)

    with tf.variable_scope('layer_four',reuse=tf.AUTO_REUSE):
        l4b_init = tf.random_normal_initializer(mean=0, stddev=0.0, dtype=tf.float32)
        l4w_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
        final_fc_weights = tf.get_variable(name="weight_four" ,
                    shape= [second_fc_output_channels, label_count],dtype=tf.float32,initializer=l4w_init)
        final_fc_bias = tf.get_variable(name="bias_four" , shape=[label_count], dtype=tf.float32,initializer=l4b_init)
        final_fc = tf.matmul(bottleneck_input, final_fc_weights) + final_fc_bias

    ground_truth_retrain_input = tf.placeholder(dtype=tf.int64, shape=[None], name=ground_truth_name)

    # The final result - a softmax can be applied to this for inference
    return final_fc, bottleneck_input,ground_truth_retrain_input





def build_graph(fingerprint_input,dropout_prob, ncep,max_len, label_count,isTraining):
    """Builds a convolutional model with low compute requirements.

    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
              v
          [Conv2D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v

    This produces slightly lower quality results than the 'conv' model, but needs
    fewer weight parameters and computations.

      During training, dropout nodes are introduced after the relu, controlled by a
      placeholder.

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """

    input_frequency_size = ncep
    input_time_size = max_len

    fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 1
    with tf.variable_scope('layer_one',reuse=tf.AUTO_REUSE):

        l1b_init = tf.random_normal_initializer(mean=0, stddev=0.1, seed=814, dtype=tf.float32)
        l1w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=814, dtype=tf.float32)


        first_weights = tf.get_variable(name="weight_one",
                    shape = [first_filter_height, first_filter_width, 1, first_filter_count],
                    dtype=tf.float32,
                    initializer=l1w_init,
                    )
        first_bias = tf.get_variable(name="bias_one", shape=[first_filter_count],dtype=tf.float32,initializer=l1b_init)
        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, first_filter_stride_y, first_filter_stride_x, 1], 'VALID') + first_bias
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
    flattened_first_conv = tf.reshape(first_dropout,[-1, first_conv_element_count])
    first_fc_output_channels = 128


    with tf.variable_scope('layer_two',reuse=tf.AUTO_REUSE):
        l2b_init = tf.random_normal_initializer(mean=0, stddev=0.1, seed=90, dtype=tf.float32)
        l2w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=90, dtype=tf.float32)

        first_fc_weights = tf.get_variable(name="weight_two",
                    shape= [first_conv_element_count, first_fc_output_channels], dtype=tf.float32,initializer=l2w_init)
        first_fc_bias = tf.get_variable(name="bias_two",shape=[first_fc_output_channels],dtype=tf.float32,initializer=l2b_init)
        first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias

        if isTraining:
            second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
        else:
            second_fc_input = first_fc

    second_fc_output_channels = 128

    with tf.variable_scope('layer_three',reuse=tf.AUTO_REUSE):
        l3b_init = tf.random_normal_initializer(mean=0, stddev=0.1, seed=666, dtype=tf.float32)
        l3w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=666, dtype=tf.float32)

        second_fc_weights = tf.get_variable(name="weight_three",
                    shape=[first_fc_output_channels, second_fc_output_channels],dtype=tf.float32,initializer=l3w_init)
        second_fc_bias = tf.get_variable(name="bias_three",shape=[second_fc_output_channels],dtype=tf.float32,initializer=l3b_init)
        second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias

        if isTraining:
            final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
        else:
            final_fc_input = second_fc

        print ('final_fc_input_shape' + str(final_fc_input.shape))


    return final_fc_input, first_weights,first_bias,first_fc_weights,first_fc_bias,second_fc_weights,second_fc_bias # The bottleneck input






def inference_frozen(ncep,max_len,label_count,nparr,frozen_graph):

    gph = load_graph(frozen_graph)
    y = gph.get_tensor_by_name('prefix/labels_softmax:0')

    fingerprint_input = gph.get_tensor_by_name('prefix/fingerprint_input:0')


    with tf.Session(graph=gph) as sess:

        y_out = sess.run(
        [
            y
        ],
        feed_dict={
            fingerprint_input: nparr
            #dropout_prob: 1.0,
        })


    return y_out


def load_graph(frozen_graph_filename):
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





def save_frozen_graph(chkpoint_dir,max_len,ncep,label_count,isTraining):


    fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    logits = build_graph(fingerprint_input=fingerprint_input, dropout_prob=dropout_prob, ncep=ncep, max_len=max_len,
                         label_count=label_count, isTraining=isTraining)

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



def inference(ncep,max_len,label_count,isTraining,nparr,chkpoint_dir):


    fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    bottleneck_input, _,_,_,_,_,_= build_graph(fingerprint_input=fingerprint_input,dropout_prob=dropout_prob,ncep=ncep,max_len=max_len,label_count=label_count,isTraining=isTraining)
    logits, _,_ = build_final_layer_graph(label_count = label_count,ncep=ncep,max_len = max_len,isTraining=isTraining,bottleneck_input=bottleneck_input)


    #checkpoint = tf.train.get_checkpoint_state(checkpoint_dir= chkpoint_dir)
    #chk_path = checkpoint.model_checkpoint_path

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #saver = tf.train.Saver(tf.global_variables())
    saver = tf.train.import_meta_graph(chkpoint_dir + 'model_labels_3.ckpt-100.meta')
    saver.restore(sess,chkpoint_dir + 'model_labels_3.ckpt-100')

    # uncomment below to debug variable names
    # between the graph in memory and the graph from checkpoint file
    # get_variable method should now reuse variables from memory scope
    # Variable method was creating new copies and suffixing the numbers to new variables in memory

    #var_name = [v.name for v in tf.global_variables()]
    #print(var_name)
    #reader = pyten.NewCheckpointReader(chk_path)
    #var_to_shape_map = reader.get_variable_to_shape_map()
    #print(var_to_shape_map)
    #print(chk_path)

    #saver.restore(sess,chk_path)
    predictions = sess.run(
        [
            logits
        ],
        feed_dict={
            fingerprint_input: nparr,
            dropout_prob: 1.0,
        })

    return predictions



def create_bottlenecks_cache(ncep,max_len,label_count,isTraining,chkpoint_dir):

    xferfiles = '/home/nitin/Desktop/tensorflow_speech_dataset/xferfiles/'
    train = '/home/nitin/Desktop/tensorflow_speech_dataset/train/'

    three_label = '/home/nitin/Desktop/tensorflow_speech_dataset/new_labels/3_labels/'


    fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    bottleneck_input,_, _, _, _, _, _ = build_graph(fingerprint_input=fingerprint_input, dropout_prob=dropout_prob, ncep=ncep, max_len=max_len,
                         label_count=label_count, isTraining=isTraining)

    print ('aiyo:' + str(bottleneck_input.shape))

    # checkpoint = tf.train.get_checkpoint_state(checkpoint_dir= chkpoint_dir)
    # chk_path = checkpoint.model_checkpoint_path

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # saver = tf.train.Saver(tf.global_variables())
    saver = tf.train.import_meta_graph(chkpoint_dir + 'model_labels_2.ckpt-30.meta', clear_devices=True)
    saver.restore(sess, chkpoint_dir + 'model_labels_2.ckpt-30')

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

    for filename in os.listdir(train):

        print ('Creating Bottleneck Inputs for file:' + filename)
        nparr = inp.prepare_file_inference(train,filename)

        nparr2 = np.reshape(nparr, [-1, max_len * ncep])

        bottleneck = sess.run(
            [
                bottleneck_input
            ],
            feed_dict={
                fingerprint_input: nparr2,
                dropout_prob: 1.0,
            })

        print ('bottleneck_shape:' + str(np.asarray(bottleneck[0]).shape))

        print('Saving Bottleneck Inputs for file:' + filename)

        np.save(xferfiles + 'numpy_bottle_' + filename +  '.npy', bottleneck)

        labels = []
        if (filename.__contains__('yes')):
            print ('yes')
            labels.append(1)
        elif (filename.__contains__('three')):
            print ('three')
            labels.append(2)
        else:
            print ('unk')
            labels.append(0)
        print('Saving Bottleneck Label for file:' + filename)

        np.save(xferfiles + 'numpy_bottle_labels_' + filename +  '.npy', np.array(labels))


def retrain(ncep,max_len,label_count,isTraining,chkpoint_dir):

    label_count = 3
    check_nans = False

    with tf.Graph().as_default() as grap:

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        bottleneck_tensor, first_weights, first_bias, first_fc_weights, first_fc_bias, second_fc_weights, second_fc_bias = build_graph(fingerprint_input=fingerprint_input, dropout_prob=dropout_prob, ncep=ncep,
                                       max_len=max_len, label_count=label_count, isTraining=isTraining)

        #print(bottleneck_tensor.shape)

        final_fc, bottleneck_input, ground_truth_retrain_input = build_final_layer_graph(label_count=label_count,ncep=ncep,max_len=max_len,isTraining=isTraining,bottleneck_input=bottleneck_tensor)


        control_dependencies = []
        if check_nans:
            checks = tf.add_check_numerics_ops()
            control_dependencies = [checks]


        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_retrain_input, logits=final_fc)

        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
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

        xferfiles_batch = '/home/nitin/Desktop/tensorflow_speech_dataset/xferfiles_batch/'

        print ('restoring')
        tf.train.Saver({'layer_one/weight_one':first_weights,'layer_one/bias_one':first_bias,'layer_two/weight_two':first_fc_weights,'layer_two/bias_two':first_fc_bias,'layer_three/weight_three':second_fc_weights,'layer_three/bias_three':second_fc_bias}).restore(sess, chkpoint_dir + 'model_labels_2.ckpt-30')

        init = tf.global_variables_initializer()
        sess.run(init)
        print ('restored')

        '''
        # Check that only part of the graph is restored 
        var_name = [v.name for v in tf.global_variables()]
        print(var_name)

        varb = grap.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print (varb)

        val = [sess.run(v) for v in varb]
        print(val)
        '''

        total_conf_matrix = None
        batch_count = 14900
        epochs = 30

        for i in range(1,epochs + 1):

            for j in range(100, batch_count, 100):

                npInputs = np.load(xferfiles_batch + 'bottleneck_batch' + '_' + str(j) + '.npy')
                npLabels = np.load(xferfiles_batch + 'bottleneck_batch_label' + '_' + str(j) + '.npy')

                print ('bottleneck input shape:' + str(npInputs.shape))
                print('bottleneck label shape:' + str(npLabels.shape))

                #npInputs2 = np.reshape(npInputs, [-1, max_len * ncep])

                xent_mean, _, conf_matrix = sess.run(
                    [
                        cross_entropy_mean, train_step,
                         confusion_matrix
                    ],
                    feed_dict={
                        #fingerprint_input: None,
                        bottleneck_input: npInputs,
                        ground_truth_retrain_input: npLabels,
                        learning_rate_input: 0.001,
                        #dropout_prob: None,
                    })

                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix

            print('epoch:' + str(i))
            print('Confusion Matrix:' + '\n' + str(total_conf_matrix))

        # Save after every 10 epochs
        #if (i % 10 == 0):
            # print('Saving checkpoint')
            # saver.save(sess=sess, save_path=chkpoint_dir + 'model_labels_' + str(label_count) + '.ckpt', global_step=i)
            # saver.export_meta_graph(filename=chkpoint_dir  + 'model_labels_' + str(label_count) + '.ckpt-' + str(i) + '.meta')

        # Validation set reporting
        #npValInputs = np.load(unk_test + 'numpy_batch_29.npy')
        #npValInputs = np.reshape(npValInputs, [-1, max_len * ncep])

        #npValLabels = np.load(unk_test + 'numpy_batch_labels_29.npy')
        #test_accuracy, conf_matrix, pred_indices = sess.run(
        #   [evaluation_step, confusion_matrix, predicted_indices],
        #    feed_dict={
        #        fingerprint_input: npValInputs,
        #        ground_truth_input: npValLabels,
        #        dropout_prob: 1.0
        #    })

        #print('Predicted Index: ' + str(pred_indices))
        #print('Actual is:' + str(npValLabels.tolist()))


#reader = pyten.NewCheckpointReader(chk_path)
    #var_to_shape_map = reader.get_variable_to_shape_map()
    #print(var_to_shape_map)
    #print(chk_path)


def train(ncep,max_len,label_count,isTraining,batch_count):

        do_bottleneck_cache  = True

        out_numpy = '/home/nitin/Desktop/tensorflow_speech_dataset/numpy/'
        unk_test = '/home/nitin/Desktop/tensorflow_speech_dataset/validate/'
        #chkpoint_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/checkpoints/'
        chkpoint_dir = '/home/nitin/PycharmProjects/habits/checkpoints/'
        predict_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/predict/'

        check_nans = False
        epochs = 30

        sess = tf.InteractiveSession()

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        #final_fc = build_graph(fingerprint_input=fingerprint_input,dropout_prob=dropout_prob,ncep=ncep,max_len = max_len,label_count=label_count,isTraining=isTraining)
        bottleneck_input = build_graph(fingerprint_input=fingerprint_input,dropout_prob=dropout_prob,ncep=ncep,max_len = max_len,label_count=label_count,isTraining=isTraining)
        final_fc,_,_ = build_final_layer_graph(label_count=label_count,isTraining=isTraining,bottleneck_input = bottleneck_input)

        # Define loss and optimizer
        ground_truth_input = tf.placeholder(
            tf.int64, [None], name='groundtruth_input')


        # Optionally we can add runtime checks to spot when NaNs or other symptoms of
        # numerical errors start occurring during training.
        control_dependencies = []
        if check_nans:
            checks = tf.add_check_numerics_ops()
            control_dependencies = [checks]


        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=final_fc)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
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



        for i in range(1,epochs + 1):

            total_conf_matrix = None

            for j in range(100,batch_count,100):


                npInputs = np.load(out_numpy + 'numpy_batch' + '_' + str(j) + '.npy')
                npLabels = np.load(out_numpy + 'numpy_batch_labels' + '_' + str(j) + '.npy'),

                npInputs2 = np.reshape(npInputs,[-1,max_len * ncep])


                xent_mean, _,_,conf_matrix = sess.run(
                    [
                        cross_entropy_mean, train_step,
                        increment_global_step,confusion_matrix
                    ],
                    feed_dict={
                        fingerprint_input: npInputs2,
                        ground_truth_input: npLabels[0],
                        learning_rate_input: 0.001,
                        dropout_prob: 0.5,
                    })

                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix

            print('epoch:' + str(i))
            print ('Confusion Matrix:' + '\n' +  str(total_conf_matrix))

            # Save after every 10 epochs
            if (i % 10 == 0):
                print('Saving checkpoint')
                saver.save(sess=sess,save_path=chkpoint_dir + 'model_labels_' + str(label_count) + '.ckpt',global_step = i )
                #saver.export_meta_graph(filename=chkpoint_dir  + 'model_labels_' + str(label_count) + '.ckpt-' + str(i) + '.meta')

            # Validation set reporting
            npValInputs = np.load(unk_test + 'numpy_batch_29.npy')
            npValInputs = np.reshape(npValInputs, [-1, max_len * ncep])

            npValLabels = np.load(unk_test + 'numpy_batch_labels_29.npy')
            test_accuracy, conf_matrix, pred_indices = sess.run(
                [evaluation_step, confusion_matrix, predicted_indices],
                feed_dict={
                    fingerprint_input: npValInputs,
                    ground_truth_input: npValLabels,
                    dropout_prob: 1.0
                })

            print('Predicted Index: ' + str(pred_indices))
            print('Actual is:' + str(npValLabels.tolist()))


def main(file_dir, file, label, label_count, chkpoint_dir):

    train(ncep=26,max_len = 99,label_count =2,isTraining=True,batch_count = 12600)
    #result = invoke_inference(file_dir=FLAGS.filedir, label= FLAGS.label, file=FLAGS.file, label_count=FLAGS.labelcount, chkpoint_dir=FLAGS.chkpoint_dir,use_graph=FLAGS.usegraph,ncep=FLAGS.ncep,max_len=FLAGS.maxlen)
    #result = invoke_inference(file_dir=file_dir, label= label, file=file, label_count=3, chkpoint_dir=chkpoint_dir,use_graph=0,ncep=26,max_len=99)
    #return result




def invoke_inference(file_dir,file,label,label_count,chkpoint_dir,use_graph,ncep,max_len):

    # hardcode isTraining for inference for now...to revisit during label automation bit..
    #ncep = 26,
    #max_len = 99
    isTraining = False

    #nparray prep
    nparr1 = inp.prepare_file_inference(file_dir = file_dir,file_name = file)
    nparr1 = np.expand_dims(nparr1,axis=0)
    nparr1 = np.reshape(nparr1,[-1,nparr1.shape[1] * nparr1.shape[2]])


    if use_graph:
        print ('Inference with graph')
        result = inference_frozen(ncep=ncep,max_len=max_len,label_count=label_count,nparr=nparr1,frozen_graph=chkpoint_dir + 'habits_frozen.pb')
    else:
        print ('Inference without graph')
        result = inference(ncep=ncep, max_len=max_len, label_count=label_count, isTraining=isTraining,nparr= nparr1,chkpoint_dir = chkpoint_dir)

    return np.argmax(result[0], axis=1)[0]

if __name__  ==   '__main__':



    file_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/predict/'
    chkpoint_dir = '/home/nitin/PycharmProjects/habits/checkpoints/'

    file = 'aud_1525770163951.wav'
    # file = 'achoo_4.wav'
    label = ''
    label_count = 3
    
    #result = main()
    #result = main(file_dir=file_dir, file=file, label=label, label_count=label_count, chkpoint_dir=chkpoint_dir)
    #print('The Result is:' + str(result))

    create_bottlenecks_cache(ncep=26,max_len = 99,label_count=2,isTraining=False,chkpoint_dir=chkpoint_dir)
    #retrain(ncep=26,max_len=99,label_count=2,isTraining=True,chkpoint_dir=chkpoint_dir)
    #retrain(ncep=26,max_len=99,label_count=3,isTraining=False,chkpoint_dir=chkpoint_dir)


    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--filedir',
            type=str,
            default='',
            help='File Directory containing the test wave file',)
    parser.add_argument(
            '--file',
            type=str,
            default='',
            help='The Test Wav File',)
    parser.add_argument(
            '--chkpoint_dir',
            type=str,
            default='',
            help='Folder path of the checkpoint files',)
    parser.add_argument(
            '--label',
            type=str,
            default='',
            help='The new label value',)
    parser.add_argument(
            '--labelcount',
            type=int,
            default=None,
            help='Number of labels ',)
    parser.add_argument(
            '--usegraph',
            type=int,
            default='',
            help='True or False, use graph',
    )
    parser.add_argument(
            '--ncep',
            type=int,
            default=26,
            help='Number of Cepstrum Coefficients to use',
    )
    parser.add_argument(
            '--maxlen',
            type=int,
            default=99,
            help='Cutoff number of mfcc frames for graph def',
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    '''






import math
import tensorflow as tf
import numpy as np

'''
def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }

'''

out_numpy = '/home/nitin/Desktop/tensorflow_speech_dataset/numpy/'
unk_test = '/home/nitin/Desktop/tensorflow_speech_dataset/unk_test/'
chkpoint_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/checkpoints/'
predict_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/predict/'


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
    first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
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
    first_fc_weights = tf.Variable(
            tf.truncated_normal(
                [first_conv_element_count, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias

    if isTraining:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc

    second_fc_output_channels = 128
    second_fc_weights = tf.Variable(
            tf.truncated_normal(
                [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias

    if isTraining:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc


    final_fc_weights = tf.Variable(
            tf.truncated_normal(
                [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

    return final_fc


def inference(ncep,max_len,label_count,isTraining):
    with tf.Session() as session:


        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        logits = build_graph(fingerprint_input=fingerprint_input,dropout_prob=dropout_prob,ncep=ncep,max_len=max_len,label_count=label_count,isTraining=isTraining)

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir= chkpoint_dir)

        chk_path = checkpoint.model_checkpoint_path
        saver.restore(session,chk_path)

        npInputs = np.load(predict_dir + 'numpy_batch_8.npy')
        #npLabels = np.load(predict_dir + 'numpy_batch_labels_1.npy'),

        npInputs2 = np.reshape(npInputs, [-1, max_len * ncep])

        predictions = session.run(
            [
                logits
            ],
            feed_dict={
                fingerprint_input: npInputs2,
                dropout_prob: 1.0,
            })

        print(predictions[0])
        print(tf.nn.softmax(predictions[0]).eval())
        predicted_indices = tf.argmax(predictions[0],axis=1)

        print(predicted_indices.eval())

        return predicted_indices





def train(ncep,max_len,label_count,isTraining,batch_count):

        check_nans = False
        epochs = 100

        sess = tf.InteractiveSession()

        fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, max_len * ncep], name="fingerprint_input")
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        final_fc = build_graph(fingerprint_input=fingerprint_input,dropout_prob=dropout_prob,ncep=ncep,max_len = max_len,label_count=label_count,isTraining=isTraining)

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
        saver = tf.train.Saver(tf.global_variables())

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

            # Validation set reporting
            npValInputs = np.load(unk_test + 'numpy_batch_12.npy')
            npValInputs = np.reshape(npValInputs, [-1, max_len * ncep])

            npValLabels = np.load(unk_test + 'numpy_batch_labels_12.npy')
            test_accuracy, conf_matrix, pred_indices = sess.run(
                [evaluation_step, confusion_matrix, predicted_indices],
                feed_dict={
                    fingerprint_input: npValInputs,
                    ground_truth_input: npValLabels,
                    dropout_prob: 1.0
                })

            print('Predicted Index: ' + str(pred_indices))
            print('Actual is:' + str(npValLabels.tolist()))




def main():

    #train(ncep=26,max_len=99,label_count=2,isTraining=True,batch_count=7700)
    inference(ncep=26,max_len=99,label_count=2,isTraining=False)


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--sample_rate',
            type=int,
            default=16000,
            help='Expected sample rate of the wavs', )
        parser.add_argument(
            '--clip_duration_ms',
            type=int,
            default=1000,
            help='Expected duration in milliseconds of the wavs', )
        parser.add_argument(
            '--clip_stride_ms',
            type=int,
            default=30,
            help='How often to run recognition. Useful for models with cache.', )
        parser.add_argument(
            '--window_size_ms',
            type=float,
            default=30.0,
            help='How long each spectrogram timeslice is', )
        parser.add_argument(
            '--window_stride_ms',
            type=float,
            default=10.0,
            help='How long the stride is between spectrogram timeslices', )
        parser.add_argument(
            '--dct_coefficient_count',
            type=int,
            default=40,
            help='How many bins to use for the MFCC fingerprint', )
        parser.add_argument(
            '--start_checkpoint',
            type=str,
            default='',
            help='If specified, restore this pretrained model before any training.')
        parser.add_argument(
            '--model_architecture',
            type=str,
            default='conv',
            help='What model architecture to use')
        parser.add_argument(
            '--wanted_words',
            type=str,
            default='yes,no,up,down,left,right,on,off,stop,go',
            help='Words to use (others will be added to an unknown label)', )
        parser.add_argument(
            '--output_file', type=str, help='Where to save the frozen graph.')
        FLAGS, unparsed = parser.parse_known_args()
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



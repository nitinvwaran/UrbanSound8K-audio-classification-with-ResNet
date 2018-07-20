import os
import numpy as np
from habits.model import AudioEventDetectionResnet as aed
import pandas as pd
import tensorflow as tf


def invoke_inference(test_batch_directory,ncep,nfft,cutoff_mfcc,cutoff_spectogram,use_nfft,batch_size,checkpoint_dir,label_count,data_format="channels_last"):

    test_count = 0

    # Read the count from the file
    if (os.path.exists(test_batch_directory)):
        with open(test_batch_directory + 'test_count.txt', 'r') as rf:
            for line in rf.readlines():
                test_count = int(line)

    os.chdir(test_batch_directory)
    print('The Test Directory is:' + str(test_batch_directory))

    with tf.Graph().as_default() as grap:
        logits, fingerprint_input, is_training = aed.build_graph(use_nfft = use_nfft,cutoff_spectogram=cutoff_spectogram,cutoff_mfcc = cutoff_mfcc,nfft = nfft,ncep = ncep,num_labels=label_count,data_format=data_format)

    with tf.Session(graph=grap) as sess:

        checkpoint_file_path = checkpoint_dir + 'urbansound8k_with_resnet.ckpt'

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

                print ('Sample np array:' + t_out[:100])

                for x in range(0,t_out.shape[0]):
                    predfile.write(t_out[x][0] + t_out[x][1] + '\n')


def accuracy(y_file):

    yfile = pd.read_csv(y_file)
    yfileActual = yfile.loc[:,'Actual']
    yfilePred = yfile.loc[:,'Prediction']

    arraycheck = np.equal(yfileActual, yfilePred)
    total_pred = np.sum(arraycheck)

    print('The test accuracy is:' + str(float(total_pred / yfile.shape[0])))






def main():

    batch_size = 250  # Could change for batch inference
    test_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/test/batch_label_count_10/'
    checkpoint_base_dir = '/home/nitin/PycharmProjects/habits/checkpoints/'
    label_meta_file_path = '/home/nitin/Desktop/tensorflow_speech_dataset/labels_meta/labels_meta.txt'
    number_cepstrums = 26
    nfft_value = 256  # Note that the FFT reduces this to n/2 + 1 as the column dimension in the spectogram matrix
    cutoff_spectogram = 75
    cutoff_mfcc = 150
    use_nfft = True

    invoke_inference(test_batch_directory=test_directory,ncep=number_cepstrums,nfft=nfft_value,cutoff_mfcc=cutoff_mfcc,
                              cutoff_spectogram=cutoff_spectogram,use_nfft=use_nfft,batch_size=batch_size,checkpoint_dir=checkpoint_base_dir,
                              label_count=10)

    accuracy(test_directory + 'ytest.txt')


if __name__ == '__main__':
    main()

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch Size', )
    parser.add_argument(
        '--test_directory',
        type=str,
        default='',
        help='Test Directory with all wav files for testing' ,)
    parser.add_argument(
        '--checkpoint_base_dir',
        type=str,
        default='',
        help='Base Directory storing all the versions of checkpoints for different graphs',
    )
    parser.add_argument(
        '--number_cepstrums',
        type=int,
        default=26,
        help='Number Cepstrums for MFCC feature engineering (Default 26)',
    )
    parser.add_argument(
        '--nfft_value',
        type=int,
        default=512,
        help='NFFT Value for Spectogram Generation (Default 512)',
    )
    parser.add_argument(
        '--label_meta_file_path',
        type=str,
        default='',
        help='Path to labels meta file',
    )
    parser.add_argument(
        '--cutoff_spectogram',
        type=int,
        default=99,
        help='Maximum number of Spectogram time slices to include during training',
    )
    parser.add_argument(
        '--cutoff_mfcc',
        type=int,
        default=99,
        help='Maximum number of MFCC frames to include during training',
    )
    parser.add_argument(
        '--use_nfft',
        type=bool,
        default=True,
        help='Flag to indicate whether spectogram or mfcc should be used (default spectogram)',
    )
    parser.add_argument(
        '--use_graph',
        type=bool,
        default=False,
        help='Flag to indicate whether to use the graph for inference or checkpoint file',
    )


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    '''


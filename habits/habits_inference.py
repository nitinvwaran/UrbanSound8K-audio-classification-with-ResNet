import os
import numpy as np
import glob
import argparse
import sys
from habits.inputs_2 import prepare_mfcc_spectogram
from habits.habits_configuration import Configuration
from habits.model import inference_frozen
from habits.model import inference
from habits.inputs_2 import get_labels_and_count


def invoke_inference(conf_object):

    os.chdir(conf_object.test_directory)
    file_name = ''
    for file in glob.glob('*.wav'):

        file_name = file
        mfcc,spectogram = prepare_mfcc_spectogram(file_dir = conf_object.test_directory,file_name = file, ncep=conf_object.ncep,nfft=conf_object.nfft
                                                  ,cutoff_mfcc=conf_object.cutoff_mfcc,cutoff_spectogram=conf_object.cutoff_spectogram
                                                  )
        if (conf_object.use_nfft):

            nparr1 = np.expand_dims(spectogram,axis=0)

        else:
            nparr1 = np.expand_dims(mfcc, axis=0)

        nparr1 = np.reshape(nparr1, [-1, nparr1.shape[1] * nparr1.shape[2]])

        print ('Shape of input matrix:' + str(nparr1.shape))


    num_labels, label_dict = get_labels_and_count(label_file=conf_object.label_meta_file_path)
    checkpoint_file_path = conf_object.checkpoint_dir + 'habits/' + 'transfer_model_label_count_' + str(num_labels) + '.ckpt'

    # What happens if base model is used?
    #checkpoint_file_path = conf_object.checkpoint_dir + 'base_dir/' + 'base_model_labels_2.ckpt-90'
    #num_labels = 2

    print ('Checkpoint File is:' + checkpoint_file_path)
    print ('File to Infer:' + file_name)

    if conf_object.use_graph:
        print ('Inference with graph')
        result = inference_frozen(nparr=nparr1,frozen_graph=conf_object + 'habits_frozen.pb') # TODO: configure once frozen graph creation automated
    else:
        print ('Inference without graph')
        result = inference(ncep=conf_object.ncep, nfft=conf_object.nfft, cutoff_mfcc = conf_object.cutoff_mfcc,cutoff_spectogram=conf_object.cutoff_spectogram,label_count=num_labels,
                           isTraining=False,nparr=nparr1,checkpoint_file_path=checkpoint_file_path,use_nfft=conf_object.use_nfft
                           )

    return np.argmax(result[0], axis=1)[0]


def main():

    '''
    
    ' Below for debugging
    
    batch_size = 1  # Could change for batch inference
    test_directory = '/home/nitin/Desktop/tensorflow_speech_dataset/predict/'
    checkpoint_base_dir = '/home/nitin/PycharmProjects/habits/checkpoints/'
    label_meta_file_path = '/home/nitin/Desktop/tensorflow_speech_dataset/labels_meta/labels_meta.txt'
    number_cepstrums = 13
    nfft_value = 512  # Note that the FFT reduces this to n/2 + 1 as the column dimension in the spectogram matrix
    cutoff_spectogram = 300
    cutoff_mfcc = 99
    use_nfft = False
    use_graph = False
    train_directory = ''
    validate_directory = ''
    train_bottleneck_dir = ''
    validate_bottleneck_dir = ''
    test_bottleneck_dir = ''
    '''

    conf_object = Configuration(test_directory=FLAGS.test_directory,
                                checkpoint_dir=FLAGS.checkpoint_base_dir, number_cepstrums=FLAGS.number_cepstrums,
                                nfft_value=FLAGS.nfft_value, label_meta_file_path=FLAS.label_meta_file_path,
                                cutoff_spectogram=FLAGS.cutoff_spectogram, cutoff_mfcc=FLAGS.cutoff_mfcc,
                                batch_size=FLAGS.batch_size, use_nfft=FLAGS.use_nfft, use_graph=FLAGS.use_graph,
                                )

    result = invoke_inference(conf_object)

    print ('The Label is:' + str(result))


if __name__ == '__main__':

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



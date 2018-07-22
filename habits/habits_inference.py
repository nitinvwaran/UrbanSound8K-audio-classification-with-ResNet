import numpy as np
from habits.model import AudioEventDetectionResnet
import pandas as pd

def invoke_inference(test_batch_directory,ncep,nfft,cutoff_mfcc, cutoff_spectogram,use_nfft,
                     batch_size,checkpoint_dir,label_count):

    aed = AudioEventDetectionResnet()
    aed.do_inference(test_batch_directory=test_batch_directory,ncep=ncep,nfft=nfft,cutoff_mfcc=cutoff_mfcc,
                     cutoff_spectogram=cutoff_spectogram,use_nfft=use_nfft,batch_size=batch_size,
                     checkpoint_dir=checkpoint_dir,label_count=label_count)


def accuracy(y_file):

    yfile = pd.read_csv(y_file)
    yfileActual = yfile.loc[:,'Actual']
    yfilePred = yfile.loc[:,'Prediction']

    arraycheck = np.equal(yfileActual, yfilePred)
    total_pred = np.sum(arraycheck)

    print('The test accuracy is:' + str(float(total_pred / yfile.shape[0])))


def main():

    batch_size = 250  # Could change for batch inference
    test_directory = '/home/ubuntu/Desktop/urbansound_data/test/batch_label_count_10/'
    checkpoint_base_dir = '/home/ubuntu/Desktop/UrbanSound8K/UrbanSound8K-audio-classification-with-ResNet/checkpoints/'
    #label_meta_file_path = '/home/ubuntu/Desktop/UrbanSound8K/UrbanSound8K-audio-classification-with-ResNet/habits/labels_meta/labels_meta.txt'
    number_cepstrums = 26
    nfft_value = 256
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


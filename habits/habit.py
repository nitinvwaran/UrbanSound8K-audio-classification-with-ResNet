import os
import shutil
from habits.inputs_2 import CommonHelpers
from habits.inputs_2 import InputRaw
from habits.model import AudioEventDetectionResnet


def create_numpy_train_batches(train_directory,regenerate_training_inputs,batch_size,
                               ncep,nfft,cutoff_mfcc,cutoff_spectogram,use_nfft):

    input_raw = InputRaw()
    lsFolds = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']

    total_count = 0

    for item in lsFolds:
        print ('The fold dir is:' + item)

        foldDir = train_directory + item + '/'


        if (regenerate_training_inputs):

            train_batch_folder, train_count = input_raw.create_numpy_batches(file_dir = foldDir,
                                                                            batch_size = batch_size,ncep=ncep,nfft=nfft,
                                                                            cutoff_mfcc = cutoff_mfcc, cutoff_spectogram=cutoff_spectogram
                                                                            ,use_nfft = use_nfft)
            with open(train_batch_folder + 'train_count.txt', 'w') as wf:
                wf.write(str(train_count) + '\n')

            total_count += train_count

        else:
            print('Re-using the existing training and validation batches')


    return total_count


def main():

    batch_size = 250

    '''
    train_directory = '/home/ubuntu/Desktop/urbansound_data/audio/'
    #validate_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/valid/'
    test_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/test/'
    checkpoint_base_dir = '/home/nitin/Desktop/aws_habits/FMSG_Habits/checkpoints/base_dir/'
    label_meta_file_path = '/home/nitin/Desktop/aws_habits/FMSG_Habits/habits/labels_meta/labels_meta.txt'
    '''


    train_directory = '/home/ubuntu/Desktop/urbansound_data/audio/'
    validate_directory = '/home/ubuntu/Desktop/urbansound_data/valid/'
    test_directory = '/home/ubuntu/Desktop/urbansound_data/test/'
    checkpoint_base_dir = '/home/ubuntu/Desktop/UrbanSound8K/UrbanSound8K-audio-classification-with-ResNet/checkpoints/'
    label_meta_file_path = '/home/ubuntu/Desktop/UrbanSound8K/UrbanSound8K-audio-classification-with-ResNet/habits/labels_meta/labels_meta.txt'


    #train_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/train_tensorboard/'
    #valid_tensorboard_dir = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/valid_tensorboard/'

    train_tensorboard_dir = '/home/ubuntu/Desktop/urbansound_data/train_tensorboard/'
    valid_tensorboard_dir = '/home/ubuntu/Desktop/urbansound_data/valid_tensorbaord/'

    do_scratch_training = True
    number_cepstrums = 26
    nfft_value = 256
    regenerate_training_inputs = False
    cutoff_spectogram = 75
    cutoff_mfcc = 150
    use_nfft = True
    num_epochs = 100
    is_training = True
    learning_rate = 0.01
    dropout_prob = 0.5

    # ResNet configurations
    data_format = 'channels_last'

    aed = AudioEventDetectionResnet()


    # Read the labels meta file
    common_helpers = CommonHelpers()
    num_labels,label_dict = common_helpers.get_labels_and_count(label_file=label_meta_file_path)
    print ('Label file data')
    print (num_labels)
    print (label_dict)

    print('Starting Scratch Training')

    if (do_scratch_training):

        print ('Starting preparing batches:')

        total_count = \
            create_numpy_train_batches(train_directory=train_directory,
                                       regenerate_training_inputs=regenerate_training_inputs,batch_size=batch_size,
                                       ncep=number_cepstrums,nfft=nfft_value,cutoff_mfcc=cutoff_mfcc,cutoff_spectogram=cutoff_spectogram,use_nfft=use_nfft
                                       )

        print ('Total count of files:' + str(total_count))

        # Start training
        print ('Starting scratch training')
        print ('Number of Labels:' + str(num_labels))

        if (os.path.exists(checkpoint_base_dir)):
            shutil.rmtree(checkpoint_base_dir)
        os.makedirs(checkpoint_base_dir)

        aed.base_train(train_folder=train_directory,validate_folder='',n_train = total_count,n_valid=total_count,
                       learning_rate=learning_rate,ncep=number_cepstrums,nfft=nfft_value,label_count=num_labels,
                       batch_size=batch_size,epochs=num_epochs,chkpoint_dir=checkpoint_base_dir,use_nfft=use_nfft,
                       cutoff_spectogram=cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,
                       data_format=data_format,train_tensorboard_dir=train_tensorboard_dir,valid_tensorboard_dir=valid_tensorboard_dir)


if __name__ == '__main__':
    main()


    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Batch Size', )
    parser.add_argument(
        '--train_directory',
        type=str,
        default='',
        help='Train Directory with all wav files for training' ,)
    parser.add_argument(
        '--validate_directory',
        type=str,
        default='',
        help='Validate Directory with all wav files for validation' ,)
    parser.add_argument(
        '--test_directory',
        type=str,
        default='',
        help='Test Directory with all wav files for testing' ,)
    parser.add_argument(
        '--train_bottleneck_dir',
        type=str,
t        default='',
        help='Bottleneck Directory for storing bottleneck files of Training Files ' ,)
    parser.add_argument(
        '--validate_bottleneck_dir',
        type=str,
        default='',
        help='Bottleneck Directory for storing bottleneck files of Validation Files' ,)
    parser.add_argument(
        '--test_bottleneck_dir',
        type=str,
        default='',
        help='Bottleneck Directory for storing bottleneck files Testing Files',
    )
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
        '--do_scratch_training',
        type=bool,
        default=False,
        help='Indicate whether training should be from scratch',
    )
    parser.add_argument(
        '--do_transfer_training',
        type=bool,
        default=False,
        help='Indicate whether training should be transfer learning',
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
        '--regenerate_training_inputs',
        type=bool,
        default=True,
        help='Flag to indicate whether training and validation inputs should be re-generated (raw data or bottlenecks)',
    )
    parser.add_argument(
        '--regenerate_test_inputs',
        type=bool,
        default=True,
        help='Flag to indicate whether test inputs should be re-generated (raw data or bottlenecks)',
    )
    parser.add_argument(
        '--use_nfft',
        type=bool,
        default=True,
        help='Flag to indicate whether spectogram or mfcc should be used (default spectogram)',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number Epochs',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning Rate',
    )
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.5,
        help='Dropout Probabilty for Regularization',
    )


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    '''




import os
import shutil
from habits.inputs_2 import CommonHelpers
from habits.inputs_2 import InputRaw
from habits.model import AudioEventDetectionResnet
import tensorflow as tf

def create_numpy_train_batches(train_directory,num_labels,validate_directory,regenerate_training_inputs,label_file,batch_size,
                               ncep,nfft,cutoff_mfcc,cutoff_spectogram,use_nfft,labels_meta):

    input_raw = InputRaw()

    # Creating the inputs - either bottleneck, or numpy arrays from scratch
    train_out_folder = train_directory + 'batch_label_count_' + str(num_labels) + '/'


    valid_out_folder = validate_directory + 'batch_label_count_' + str(num_labels) + '/'

    train_count = 0
    valid_count = 0

    if (os.path.exists(train_out_folder)):
        with open(train_out_folder + 'train_count.txt', 'r') as rf:
            for line in rf.readlines():
                train_count = int(line)

    if (os.path.exists(valid_out_folder)):
        with open(valid_out_folder + 'valid_count.txt', 'r') as rf:
            for line in rf.readlines():
                valid_count = int(line)


    if (regenerate_training_inputs):

        # Generate numpy files for train and validate
        # Existing numpy batches will be erased and replaced
        try:

            train_out_folder, train_count = input_raw.create_numpy_batches(file_dir = train_directory,label_count=num_labels,
                                                                           label_file= label_file,batch_size = batch_size,ncep=ncep,nfft=nfft,
                                                                           cutoff_mfcc = cutoff_mfcc, cutoff_spectogram=cutoff_spectogram
                                                                           ,use_nfft = use_nfft,labels_meta=labels_meta)
            with open(train_out_folder + 'train_count.txt', 'w') as wf:
                wf.write(str(train_count) + '\n')

            # Create validate batches
            valid_out_folder, valid_count = input_raw.create_numpy_batches(file_dir = validate_directory,label_count=num_labels,
                                                                           label_file= label_file,batch_size = batch_size,ncep=ncep,nfft=nfft,
                                                                           cutoff_mfcc = cutoff_mfcc, cutoff_spectogram=cutoff_spectogram
                                                                           ,use_nfft = use_nfft,labels_meta=labels_meta)

            with open(valid_out_folder + 'valid_count.txt', 'w') as wf:
                wf.write(str(valid_count) + '\n')

        except Exception as e:
            if (os.path.exists(train_out_folder)):
                shutil.rmtree(train_out_folder)

            if (os.path.exists(valid_out_folder)):
                shutil.rmtree(valid_out_folder)

            raise e


    else:
        print('Re-using the existing training and validation batches')


    return train_out_folder,valid_out_folder,train_count,valid_count


def create_numpy_test_batches(test_directory,num_labels,regenerate_test_inputs,label_file,batch_size,
                               ncep,nfft,cutoff_mfcc,cutoff_spectogram,use_nfft,labels_meta):

    input_raw = InputRaw()

    if (test_directory.strip() == ''):
        print ('You must specify a test directory with wav files for testing, my son')
        raise Exception('No Test Directory Specified!')


    # Creating the inputs - either bottleneck, or numpy arrays from scratch
    test_out_folder = test_directory + 'batch_label_count_' + str(num_labels) + '/'

    test_count = 0
    if (os.path.exists(test_out_folder)):
        with open(test_out_folder + 'test_count.txt', 'r') as rf:
            for line in rf.readlines():
                test_count = int(line)

        # Create test batches
        # Existing numpy batches will be erased and replaced
    if (regenerate_test_inputs):

        try:
            test_out_folder, test_count = input_raw.create_numpy_batches(file_dir = test_directory,label_count=num_labels
                                                                         ,label_file=label_file,batch_size=batch_size,ncep=ncep,
                                                                         nfft=nfft,cutoff_mfcc=cutoff_mfcc,cutoff_spectogram=cutoff_spectogram,
                                                                         use_nfft=use_nfft,labels_meta=labels_meta)
            with open (test_out_folder + 'test_count.txt','w') as wf:
                wf.write(str(test_count) + '\n')

        except Exception as e:
            if (os.path.exists(test_out_folder)):
                shutil.rmtree(test_out_folder)

            raise e

    else:
        print ('Not generating test batches, reusing where applicable')

    return test_out_folder,test_count




def run_validations(label_meta_file_path,do_scratch_training, train_directory, validate_directory):

    if (label_meta_file_path.strip() == ''):
        print('You must specify a label meta file')
        raise Exception('You must specify a label meta file')
    elif (do_scratch_training):
        if(train_directory.strip() == ''):
            print ('You must specify a training directory with wav files for scratch training')
            raise Exception('You must specify a training directory with wav files for scratch training')
        elif (validate_directory.strip() == ''):
            print('You must specify a validation directory with wav files for scratch training')
            raise Exception('You must specify a validation directory with wav files for scratch training')

def main():

    batch_size = 500
    train_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/train/'
    validate_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/valid/'
    test_directory = '/home/nitin/Desktop/sdb1/all_files/tensorflow_voice/UrbanSound8K/test/'
    checkpoint_base_dir = '/home/nitin/Desktop/aws_habits/FMSG_Habits/checkpoints/base_dir/'
    label_meta_file_path = '/home/nitin/Desktop/aws_habits/FMSG_Habits/habits/labels_meta/labels_meta.txt'

    do_scratch_training = True
    number_cepstrums = 13
    nfft_value = 512  # Note that the FFT reduces this to n/2 + 1 as the column dimension in the spectogram matrix
    regenerate_training_inputs = False
    regenerate_test_inputs = False
    cutoff_spectogram = 300
    cutoff_mfcc = 99
    use_nfft = True
    num_epochs = 30
    is_training = True
    learning_rate = 0.001
    dropout_prob = 0.5


    # ResNet configurations
    resnet_size = 18
    bottleneck = False
    num_classes = 10
    num_filters = 64
    kernel_size = 7
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    block_sizes = [2,2,2,2] # for resnet of 18 layers
    block_strides = [1,2,2,2] # General for all block sizes
    final_size = 512 # no bottleneck
    resnet_version = 2
    data_format = 'channels_last'
    dtype = tf.float32

    aed = AudioEventDetectionResnet()

    run_validations(label_meta_file_path=label_meta_file_path,do_scratch_training=do_scratch_training,train_directory=train_directory,
                    validate_directory=validate_directory)

    # Read the labels meta file
    common_helpers = CommonHelpers()
    num_labels,label_dict = common_helpers.get_labels_and_count(label_file=label_meta_file_path)
    print ('Label file data')
    print (num_labels)
    print (label_dict)

    print('Starting Scratch Training')

    if (do_scratch_training):

        print ('Starting preparing batches:')

        train_out_folder, valid_out_folder, train_count, valid_count = \
            create_numpy_train_batches(train_directory=train_directory,num_labels=num_labels,validate_directory=validate_directory,
                                       regenerate_training_inputs=regenerate_training_inputs,label_file=label_meta_file_path,batch_size=batch_size,
                                       ncep=number_cepstrums,nfft=nfft_value,cutoff_mfcc=cutoff_mfcc,cutoff_spectogram=cutoff_spectogram,use_nfft=use_nfft,
                                       labels_meta=label_dict)
        test_out_folder, test_count = create_numpy_test_batches(test_directory=test_directory,num_labels = num_labels,regenerate_test_inputs=regenerate_test_inputs,
                                                                label_file=label_meta_file_path, batch_size=batch_size,
                                                                ncep=number_cepstrums, nfft=nfft_value,
                                                                cutoff_mfcc=cutoff_mfcc,
                                                                cutoff_spectogram=cutoff_spectogram, use_nfft=use_nfft,
                                                                labels_meta=label_dict
                                                                )


        # Start training
        print ('Starting scratch training')
        print ('Params are; train folder: ' + str(train_out_folder) + ' valid folder: ' + str(valid_out_folder) + ' number train files: ' + str(train_count) + ' number validation files:  ' + str(valid_count))
        print ('Number of Labels:' + str(num_labels))

        #nfft = int(conf_object.nfft / 2 + 1)

        #if (conf_object.use_nfft):
        #    max_len = conf_object.cutoff_spectogram
        #else:
        #    max_len = conf_object.cutoff_mfcc

        # Base  checkpoint directory which will always store only 1 model from scratch
        # TODO: create freeze graph version of the scratch model

        if (os.path.exists(checkpoint_base_dir)):
            shutil.rmtree(checkpoint_base_dir)
        os.makedirs(checkpoint_base_dir)

        aed.base_train(train_folder=train_out_folder,validate_folder=valid_out_folder,n_train = train_count,n_valid=valid_count,
                       learning_rate=learning_rate,dropoutprob=dropout_prob,ncep=number_cepstrums,nfft=nfft_value,label_count=num_labels,
                       isTraining=is_training,batch_size=batch_size,epochs=num_epochs,chkpoint_dir=checkpoint_base_dir,use_nfft=use_nfft,
                       cutoff_spectogram=cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,bottleneck=bottleneck,num_filters=num_filters,kernel_size = kernel_size
                       ,conv_stride = conv_stride,first_pool_stride = first_pool_stride,first_pool_size=first_pool_size,block_sizes = block_sizes,final_size = final_size,
                       resnet_version = resnet_version,data_format=data_format)


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




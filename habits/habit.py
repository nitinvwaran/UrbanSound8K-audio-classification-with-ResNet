import os
import shutil
from habits.inputs_2 import CommonHelpers
from habits.inputs_2 import InputRaw
from habits.habits_configuration import Configuration
from habits.model import AudioEventDetection


def create_numpy_train_batches(conf_object):

    input_raw = InputRaw()

    # Creating the inputs - either bottleneck, or numpy arrays from scratch
    train_out_folder = conf_object.train_directory + 'batch_label_count_' + str(conf_object.num_labels) + '/'
    valid_out_folder = conf_object.validate_directory + 'batch_label_count_' + str(conf_object.num_labels) + '/'

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


    if (conf_object.regenerate_training_inputs):


        # Generate numpy files for train and validate
        # Existing numpy batches will be erased and replaced
        try:

            train_out_folder, train_count = input_raw.create_numpy_batches(conf_object)
            with open(train_out_folder + 'train_count.txt', 'w') as wf:
                wf.write(str(train_count) + '\n')

            # Create validate batches
            valid_out_folder, valid_count = input_raw.create_numpy_batches(conf_object)

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


def create_numpy_test_batches(conf_object):

    input_raw = InputRaw()

    if (conf_object.test_directory.strip() == ''):
        print ('You must specify a test directory with wav files for testing, my son')
        raise Exception('No Test Directory Specified!')


    # Creating the inputs - either bottleneck, or numpy arrays from scratch
    test_out_folder = conf_object.test_directory + 'batch_label_count_' + str(conf_object.num_labels) + '/'

    test_count = 0
    if (os.path.exists(test_out_folder)):
        with open(test_out_folder + 'test_count.txt', 'r') as rf:
            for line in rf.readlines():
                test_count = int(line)

        # Create test batches
        # Existing numpy batches will be erased and replaced
    if (conf_object.regenerate_test_inputs):

        try:
            test_out_folder, test_count = input_raw.create_numpy_batches(conf_object)
            with open (test_out_folder + 'test_count.txt','w') as wf:
                wf.write(str(test_count) + '\n')

        except Exception as e:
            if (os.path.exists(test_out_folder)):
                shutil.rmtree(test_out_folder)

            raise e

    else:
        print ('Not generating test batches, reusing where applicable')

    return test_out_folder,test_count


def create_train_bottleneck_batches(conf_object):

    aed = AudioEventDetection()
    input_raw = InputRaw()

    bottleneck_batches_train_dir = conf_object.train_bottleneck_dir + 'batch_label_count_' + str(conf_object.num_labels) + '/'
    bottleneck_batched_valid_dir = conf_object.validate_bottleneck_dir + 'batch_label_count_' + str(conf_object.num_labels) + '/'

    train_files_count = 0
    if (os.path.exists(bottleneck_batches_train_dir)):
        with open(bottleneck_batches_train_dir + 'train_count.txt', 'r') as rf:
            for line in rf.readlines():
                train_files_count = int(line)

    valid_file_count = 0
    if (os.path.exists(bottleneck_batched_valid_dir)):
        with open(bottleneck_batched_valid_dir + 'valid_count.txt', 'r') as rf:
            for line in rf.readlines():
                valid_file_count = int(line)

    if (conf_object.regenerate_training_inputs): # No point bottlenecking batches for Test - direct inference on the new graph version

        base_checkpoint_dir = conf_object.checkpoint_dir + 'base_dir/'

        print ('Starting Transfer Training')

        print ('Base Checkpoint Dir:' + base_checkpoint_dir)
        print ('Num_Labels is:' + str(conf_object.num_labels))

        print ('Creating bottleneck cache files for train data in folder:' + conf_object.train_bottleneck_dir)

        try:

            train_files_count = aed.create_bottlenecks_cache(file_dir=conf_object.train_directory,bottleneck_input_dir=conf_object.train_bottleneck_dir,conf_object = conf_object)



            print('Creating bottleneck cache batch for train data')
            bottleneck_batches_train_dir = input_raw.create_randomized_bottleneck_batches(file_dir = conf_object.train_bottleneck_dir,conf_object = conf_object)
            print('Created bottleneck cache batch for train data in folder:' + bottleneck_batches_train_dir)

            with open (bottleneck_batches_train_dir + 'train_count.txt','w') as wf:
                wf.write(str(train_files_count) + '\n')

            print('Creating bottleneck cache files for validation data in folder:' + conf_object.validate_bottleneck_dir)

            valid_file_count = aed.create_bottlenecks_cache(file_dir = conf_object.validate_directory,bottleneck_input_dir = conf_object.validate_bottleneck_dir,conf_object = conf_object)

            print('Creating bottleneck cache batch for train data')
            bottleneck_batched_valid_dir = input_raw.create_randomized_bottleneck_batches(file_dir = conf_object.validate_bottleneck_dir,conf_object = conf_object)
            print('Created bottleneck cache batch for validation data in folder:' + bottleneck_batched_valid_dir)


            with open (bottleneck_batched_valid_dir + 'valid_count.txt','w') as wf:
                wf.write(str(valid_file_count) + '\n')

        except Exception as e:
            if (os.path.exists(bottleneck_batches_train_dir)):
                shutil.rmtree(bottleneck_batches_train_dir)

            if (os.path.exists(bottleneck_batched_valid_dir)):
                shutil.rmtree(bottleneck_batched_valid_dir)

            raise e

    else:
        print ('Reusing Existing Bottleneck Batches')

    return bottleneck_batches_train_dir,bottleneck_batched_valid_dir,train_files_count,valid_file_count


def run_validations(conf_object):

    if (not conf_object.do_transfer_training and not conf_object.do_scratch_training):
        print ('You must either do transfer or scratch learning..you cannot do neither')
        raise Exception('You must either do transfer or scratch learning...you cannot do neither')
    elif (conf_object.do_transfer_training and conf_object.do_scratch_training):
        print ('You cannot do both scratch and transfer learning simultaneously')
        raise Exception('You cannot do both scratch and transfer learning simultaneously')
    elif (conf_object.label_meta_file_path.strip() == ''):
        print('You must specify a label meta file')
        raise Exception('You must specify a label meta file')
    elif (conf_object.do_scratch_training):
        if(conf_object.train_directory.strip() == ''):
            print ('You must specify a training directory with wav files for scratch training')
            raise Exception('You must specify a training directory with wav files for scratch training')
        elif (conf_object.validate_directory.strip() == ''):
            print('You must specify a validation directory with wav files for scratch training')
            raise Exception('You must specify a validation directory with wav files for scratch training')
    elif (conf_object.do_transfer_training):
        if (conf_object.train_bottleneck_dir.strip() == ''):
            print('You must specify a training directory with wav files for transfer learning')
            raise Exception('You must specify a training directory with wav files for transfer learning')
        if (conf_object.validate_bottleneck_dir.strip() == ''):
            print('You must specify a validation directory with wav files for transfer learning')
            raise Exception('You must specify a validation directory with wav files for transfer learning')


def main():

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' # This code block to go in console call version '
    ' # If more config to add, add them to the configuration class, then the command line parser below, then here. '

    'conf_object = Configuration(train_directory=FLAGS.train_directory,validate_directory=FLAGS.validate_directory,test_directory=FLAGS.test_directory,train_bottleneck_dir=FLAGS.train_bottleneck_dir,'
    '                   validate_bottleneck_dir=FLAGS.validate_bottleneck_dir,test_bottleneck_dir = FLAGS.test_bottleneck_dir,'
    '                   checkpoint_dir=FLAGS.checkpoint_base_dir,number_cepstrums=FLAGS.number_cepstrums,nfft_value=FLAGS.nfft_value,label_meta_file_path=FLAGS.label_meta_file_path,'
    '                   do_scratch_training=FLAGS.do_scratch_training,do_transfer_training=FLAGS.do_transfer_training, cutoff_spectogram = FLAGS.cutoff_spectogram,cutoff_mfcc=FLAGS.cutoff_mfcc,'
    '                   regenerate_training_inputs =FLAGS.regenerate_training_inputs,regenerate_test_inputs=FLAGS.regenerate_test_inputs,batch_size=FLAGS.batch_size,use_nfft = FLAGS.use_nfft'
    '                   ,num_epochs = FLAGS.num_epochs,learning_rate = FLAGS.learning_rate,dropout_prob = FLAGS.dropout_prob)'
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    batch_size = 1000
    train_directory = '/home/ubuntu/Desktop/vggish/train/'
    validate_directory = '/home/ubuntu/Desktop/vggish/valid/'
    test_directory = '/home/ubuntu/Desktop/vggish/test'
    train_bottleneck_dir = '/home/ubuntu/Desktop/vggish/xferfiles/'
    validate_bottleneck_dir = '/home/ubuntu/Desktop/vggish/xferfiles_valid/'
    test_bottleneck_dir = 'x'
    checkpoint_base_dir = '/home/ubuntu/Desktop/aws_habits/FMSG_Habits/checkpoints/'
    label_meta_file_path = '/home/ubuntu/Desktop/aws_habits/FMSG_Habits/habits/labels_meta/labels_meta.txt'
    do_scratch_training = True
    do_transfer_training = False
    number_cepstrums = 13
    nfft_value = 512 # Note that the FFT reduces this to n/2 + 1 as the column dimension in the spectogram matrix
    regenerate_training_inputs = False
    regenerate_test_inputs = False
    cutoff_spectogram = 300
    cutoff_mfcc = 99
    use_nfft = False
    num_epochs = 100
    is_training = True

    conf_object = Configuration(train_directory=train_directory,validate_directory=validate_directory,test_directory=test_directory,train_bottleneck_dir=train_bottleneck_dir,
                       validate_bottleneck_dir=validate_bottleneck_dir,test_bottleneck_dir = test_bottleneck_dir,
                       checkpoint_dir=checkpoint_base_dir,number_cepstrums=number_cepstrums,nfft_value=nfft_value,label_meta_file_path=label_meta_file_path,
                       do_scratch_training=do_scratch_training,do_transfer_training=do_transfer_training, cutoff_spectogram = cutoff_spectogram,cutoff_mfcc=cutoff_mfcc,
                       regenerate_training_inputs =regenerate_training_inputs,regenerate_test_inputs=regenerate_test_inputs,batch_size=batch_size,use_nfft=use_nfft
                        ,num_epochs = num_epochs
                        )

    aed = AudioEventDetection()

    run_validations(conf_object)

    # Read the labels meta file
    common_helpers = CommonHelpers()
    num_labels,label_dict = common_helpers.get_labels_and_count(label_file=conf_object.label_meta_file_path)

    conf_object.num_labels = num_labels
    conf_object.labels_dict = label_dict

    if (conf_object.do_scratch_training):

        train_out_folder, valid_out_folder, train_count, valid_count = create_numpy_train_batches(conf_object = conf_object)
        test_out_folder, test_count = create_numpy_test_batches(conf_object=conf_object)


        # Start training
        print ('Starting scratch training')
        print ('Params are; train folder: ' + str(train_out_folder) + ' valid folder: ' + str(valid_out_folder) + ' number train files: ' + str(train_count) + ' number validation files:  ' + str(valid_count))
        print ('Number of Labels:' + str(num_labels))

        nfft = int(conf_object.nfft / 2 + 1)

        if (conf_object.use_nfft):
            max_len = conf_object.cutoff_spectogram
        else:
            max_len = conf_object.cutoff_mfcc

            # Base  checkpoint directory which will always store only 1 model from scratch
            # TODO: create freeze graph version of the scratch model
            base_checkpoint_dir = conf_object.checkpoint_dir + 'base_dir/'

            if (os.path.exists(base_checkpoint_dir)):
                shutil.rmtree(base_checkpoint_dir)
            os.makedirs(base_checkpoint_dir)

            aed.base_train(train_folder=train_out_folder,validate_folder=valid_out_folder,n_train = train_count,n_valid=valid_count,conf_object = conf_object)

    if (conf_object.do_transfer_training):

        bottleneck_batches_train_dir, bottleneck_batched_valid_dir, train_files_count, \
        valid_file_count = create_train_bottleneck_batches(conf_object=conf_object)

        # Retrain the weights for just the softmax layer, given the bottleneck inputs
        aed.retrain(train_bottleneck_dir=bottleneck_batches_train_dir,
                      valid_bottleneck_dir = bottleneck_batched_valid_dir,n_count = train_files_count,n_valid_count = valid_file_count,
                      conf_object = conf_object
                      )





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




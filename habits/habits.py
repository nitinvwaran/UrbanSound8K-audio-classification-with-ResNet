import habits.model as model
import habits.inputs_2 as input
import tensorflow as tf
import sys
from habits.config import Configuration as conf
import argparse



def main(_):

    # If more config to add, add them to the configuration class, then the command line parser below, then here.
    conf_object = conf(train_directory=FLAGS.train_directory,validate_directory=FLAGS.validate_directory,test_directory=FLAGS.test_directory,train_bottleneck_dir=FLAGS.train_bottleneck_dir,
                       validate_bottleneck_dir=FLAGS.validate_bottleneck_dir,test_bottleneck_dir = FLAGS.test_bottleneck_dir,
                       checkpoint_dir=FLAGS.checkpoint_base_dir,number_cepstrums=FLAGS.number_cepstrums,nfft_value=FLAGS.nfft_value,label_meta_file_path=FLAGS.label_meta_file_path,
                       do_scratch_training=FLAGS.do_scratch_training,do_transfer_training=FLAGS.do_transfer_training, cutoff_spectogram = FLAGS.cutoff_spectogram,cutoff_mfcc=FLAGS.cutoff_mfcc,
                       regenerate_training_inputs =FLAGS.regenerate_training_inputs,regenerate_test_inputs=FLAGS.regenerate_test_inputs,batch_size=FLAGS.batch_size
                       )

    if (conf_object.do_transfer_training == 0 and conf_object.do_transfer_training == 0):
        print ('You must either do transfer or scratch learning, my son...you cannot do neither')
        return -1
    elif (conf_object.do_transfer_training == 1 and conf_object.do_scratch_training == 1):
        print ('You cannot do both scratch and transfer learning simultaneously, my son')
        return -1
    elif (conf_object.label_meta_file_path.strip() == ''):
        print('You must specify a label meta file, my son')
        return -1
    elif (conf_object.do_scratch_training == 1):
        if(conf_object.train_directory.strip() == ''):
            print ('You must specify a training directory with wav files for scratch training, my son')
            return -1
        elif (conf_object.validate_directory.strip() == ''):
            print('You must specify a validation directory with wav files for scratch training, my son')
            return -1
    elif (conf_object.do_transfer_training == 1):
        if (conf_object.train_bottleneck_dir.strip() == ''):
            print('You must specify a training directory with wav files for transfer learning, my son')
            return -1
        if (conf_object.validate_bottleneck_dir.strip() == ''):
            print('You must specify a validation directory with wav files for transfer learning, my son')
            return -1


    # Read the labels meta file
    num_labels,label_dict = input.get_labels_and_count(conf_object.label_meta_file_path)


    # Creating the inputs - either bottleneck, or numpy arrays from scratch
    if (conf_object.do_scratch_training == 1):
        if (conf_object.regenerate_training_inputs == 1):

            # Generate numpy files for train and validate
            # Existing numpy batches will be erased and replaced
            input.create_numpy_batches(file_dir= conf_object.train_directory,out_dir=conf_object.train_directory,label_count=num_labels,
                                       label_file = conf_object.label_meta_file_path,cutoff_mfcc=conf_object.cutoff_mfcc,
                                       cutoff_spectogram=conf_object.cutoff_spectogram,batch_size=conf_object.batch_size,ncep=conf_object.ncep,
                                       nfft=conf_object.nfft
                                       )
            # Create validate batches
            input.create_numpy_batches(file_dir=conf_object.validate_directory, out_dir=conf_object.validate_directory,
                                       label_count=num_labels,
                                       label_file=conf_object.label_meta_file_path, cutoff_mfcc=conf_object.cutoff_mfcc,
                                       cutoff_spectogram=conf_object.cutoff_spectogram,
                                       batch_size=conf_object.batch_size, ncep=conf_object.ncep,
                                       nfft=conf_object.nfft
                                       )
        else:
            print ('Re-using the existing training abd validation batches')
            print ("\n")
            print ("\n")


        if (conf_object.regenerate_test_inputs == 1):
            if (conf_object.test_directory.strip() == ''):
                print ('You must specify a test directory with wav files for scratch training, my son')
                return -1

            # Create test batches
            # Existing numpy batches will be erased and replaced
            input.create_numpy_batches(file_dir=conf_object.test_directory,
                                        out_dir=conf_object.test_directory,
                                        label_count=num_labels,
                                        label_file=conf_object.label_meta_file_path,
                                        cutoff_mfcc=conf_object.cutoff_mfcc,
                                        cutoff_spectogram=conf_object.cutoff_spectogram,
                                        batch_size=conf_object.batch_size, ncep=conf_object.ncep,
                                        nfft=conf_object.nfft
                                        )



        else:
            print ('Not generating test batches')



if __name__ == '__main__':

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
        default='',
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
        type=int,
        default=0,
        help='Indicate whether training should be from scratch',
    )
    parser.add_argument(
        '--do_transfer_training',
        type=int,
        default=0,
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
        type=int,
        default=1,
        help='Flag to indicate whether training and validation inputs should be re-generated (raw data or bottlenecks)',
    )
    parser.add_argument(
        '--regenerate_test_inputs',
        type=int,
        default=1,
        help='Flag to indicate whether test inputs should be re-generated (raw data or bottlenecks)',
    )


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


    '''
    file_dir = '/home/nitin/Desktop/tensorflow_speech_dataset/predict/'
    chkpoint_dir = '/home/nitin/PycharmProjects/habits/checkpoints/'

    file = 'aud_1525770163951.wav'
    # file = 'achoo_4.wav'
    label = ''
    label_count = 3

    #result = main()
    #result = main(file_dir=file_dir, file=file, label=label, label_count=label_count, chkpoint_dir=chkpoint_dir)
    #print('The Result is:' + str(result))

    #create_bottlenecks_cache(ncep=26,max_len = 99,label_count=2,isTraining=False,chkpoint_dir=chkpoint_dir)
    #retrain(ncep=26,max_len=99,label_count=2,isTraining=True,chkpoint_dir=chkpoint_dir)
    retrain(ncep=26,max_len=99,label_count=3,isTraining=False,chkpoint_dir=chkpoint_dir)

    '''





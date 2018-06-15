class Configuration(object):
    def __init__(self, train_directory, validate_directory, test_directory, train_bottleneck_dir,
                 validate_bottleneck_dir, test_bottleneck_dir, checkpoint_dir, number_cepstrums, nfft_value,
                 label_meta_file_path, do_scratch_training=False, do_transfer_training=False, cutoff_spectogram=99,
                 cutoff_mfcc=99, regenerate_training_inputs=False, regenerate_test_inputs=False, batch_size=1000,use_nfft =True
                 ,num_epochs=20,learning_rate = 0.001,dropout_prob = 0.5,use_graph=False,num_labels = -1,labels_dict = {},is_training=True):
        self.num_labels = num_labels
        self.train_directory = train_directory
        self.validate_directory = validate_directory
        self.test_directory = test_directory

        self.train_bottleneck_dir = train_bottleneck_dir
        self.validate_bottleneck_dir = validate_bottleneck_dir
        self.test_bottleneck_dir = test_bottleneck_dir

        self.checkpoint_dir = checkpoint_dir
        self.ncep = number_cepstrums
        self.nfft = nfft_value
        self.label_meta_file_path = label_meta_file_path

        self.cutoff_spectogram = cutoff_spectogram
        self.cutoff_mfcc = cutoff_mfcc
        self.do_scratch_training = do_scratch_training
        self.do_transfer_training = do_transfer_training
        self.regenerate_training_inputs = regenerate_training_inputs
        self.regenerate_test_inputs = regenerate_test_inputs
        self.batch_size = batch_size
        self.use_nfft = use_nfft
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.use_graph = use_graph
        self.labels_dict = labels_dict

        self.is_training = is_training



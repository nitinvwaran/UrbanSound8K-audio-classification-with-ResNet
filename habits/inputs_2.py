import glob
import os
import scipy.io.wavfile as wav
import python_speech_features as pspeech
import numpy as np
import scipy.signal as sig
import shutil


#class InputsVGG(object):
#    def






class CommonHelpers(object):

    def get_labels_and_count(self,label_file):

        dict_labels = {}
        num_labels = 0
        with open (label_file,'r') as labelfile:
            data = labelfile.readlines()

            for line in data:
                num_labels = num_labels + 1
                dict_labels.update({int(line.split(':')[0].strip()) : line.split(':')[1].strip().lower()})

        return (num_labels, dict_labels)

    def reset_folder_make_new(self,file_dir, label_count):

        version_out_dir = file_dir + 'batch_label_count_' + str(label_count) + '/'

        # Make a new directory, if this is a new graph version, to store all the batches in there for the new graph version
        if (os.path.isdir(version_out_dir)):
            shutil.rmtree(version_out_dir)

        os.makedirs(version_out_dir)

        return version_out_dir

    def stamp_label(self,num_labels, labels_meta, filename):
        # Brands the file with one of x labels
        # The filename must contain the string label name in lowercase
        l = 0  # default
        for j in range(0, num_labels):
            if (filename.__contains__(labels_meta[j])):
                l = j  # Assign the label and break, overrides default
                break

        return l

class InputRaw(object):

    def prepare_mfcc_spectogram(self,file_dir,file_name,ncep,nfft,cutoff_mfcc,cutoff_spectogram,mfcc_padding_value = 0,specto_padding_value = 0):


        fs,signal = wav.read(file_dir + file_name)
        mfcc = pspeech.mfcc(signal=signal,samplerate=fs,numcep=ncep)
        f, t, specgram = sig.spectrogram(x=signal,fs=fs,nfft=nfft)

        # Truncate mfcc frames to specified maximum cutoff
        if(mfcc.shape[0] > cutoff_mfcc):
                mfcc = mfcc[:cutoff_mfcc,:]

        # MFCC: Apply padding if frame length lower than cutoff
        mfcc_padding = ((0, cutoff_mfcc - mfcc.shape[0]), (0, 0))
        nparr_mfcc = np.pad(mfcc, pad_width=mfcc_padding, mode='constant', constant_values=mfcc_padding_value) # Pad the mfcc with the padding value

        # Spectogram padding
        specgram2 = specgram.transpose() # Time major
        if (specgram2.shape[0] > cutoff_spectogram):
            specgram2 = specgram2[:cutoff_spectogram,:]

        specgram_padding = ((0,cutoff_spectogram - specgram2.shape[0]),(0,0))
        nparr_specgram = np.pad(specgram2,pad_width=specgram_padding,mode='constant',constant_values=specto_padding_value) # Pad with input spectogram value

        return nparr_mfcc,nparr_specgram



    def create_randomized_bottleneck_batches(self,file_dir,label_count,batch_size):

        common_helpers = CommonHelpers()
        os.chdir(file_dir)
        file_dir_out = common_helpers.reset_folder_make_new(file_dir = file_dir,label_count=label_count)

        # TODO: Randomize, though this selection 'should' be random...
        file_count = int(len([name for name in os.listdir('.') if os.path.isfile(name)]) / 2)

        print('Count of bottleneck files in  directory' + file_dir + ' is: ' + str(file_count))
        print('Preparing the numpy bottleneck batches to the directory:' + file_dir_out)

        inputs = []
        labels = []

        i = 0
        for file in glob.glob('*.npy'):

            if (file.__contains__('label')):
                file2 = file.replace('numpy_bottle_labels_','')
                nparr = np.load(file)
                labels.append(nparr.tolist())

                input_file = 'numpy_bottle_' + file2
                nparr2 = np.load(input_file)
                inputs.append(nparr2.tolist())
                i = i + 1

            if ((len(labels) != 0 and len(labels) % batch_size == 0) or i == file_count):
                print ('Creating Bottleneck Batch:' + str(i))
                np.save(file_dir_out + 'bottleneck_batch_' + str(i) + '.npy',np.asarray(inputs))
                np.save(file_dir_out + 'bottleneck_batch_label_' + str(i) + '.npy',np.asarray(labels))

                inputs = []
                labels = []

        return file_dir_out

    def create_numpy_batches(self,file_dir,out_dir, label_count, label_file, cutoff_mfcc, cutoff_spectogram, batch_size = 500, ncep = 13, nfft = 512,use_nfft = True):

        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        ' Creates numpy batches for scratch training '
        ' A new batch can be created for each graph version '
        ' Should only be needed for baselining, given bottlenecks are being used for xfer learning'
        ' bottlenecks are created using another method'
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        common_helpers = CommonHelpers()
        os.chdir(file_dir)
        file_count = len([name for name in os.listdir('.') if os.path.isfile(name)])
        if (file_count == 0):
            print ('No files in the directory:' + file_dir + ' exiting now')
            raise Exception ('No files in the directory:' + file_dir)

        i = 0
        inputs = []
        labels = []
        num_labels, labels_meta = common_helpers.get_labels_and_count(label_file)
        version_out_dir = common_helpers.reset_folder_make_new(out_dir, label_count)

        print ('Count of files in training directory' + file_dir + ' is: ' + str(file_count))
        print ('Preparing the numpy batches to the directory:' + version_out_dir)

        for file in glob.glob("*.wav"):

            mfcc,spectogram = self.prepare_mfcc_spectogram(file_dir = file_dir,file_name=file,ncep=ncep,nfft=nfft,cutoff_mfcc=cutoff_mfcc,cutoff_spectogram=cutoff_spectogram)

            if (use_nfft):
                input_raw = spectogram.tolist()

            else:
                input_raw = mfcc.tolist()

            inputs.append(input_raw)

            l = common_helpers.stamp_label(num_labels=num_labels,labels_meta=labels_meta,filename=file)
            labels.append(l)

            i = i + 1

            if (i % batch_size == 0 or i == file_count):

                npInputs = np.array(inputs)
                npLabels = np.array(labels)

                print ('Saving batch ' + str(i) + ' to the output dir ' + version_out_dir)
                # Numpy batch dump the voice files in batches of batch_size
                np.save(version_out_dir + 'models_label_count_' + str(label_count) + '_numpy_batch' + '_' + str(i) + '.npy',npInputs)
                np.save(version_out_dir + 'models_label_count_' + str(label_count) + '_numpy_batch_labels' + '_' + str(i) + '.npy', npLabels)
                inputs = []
                labels = []

        return version_out_dir, file_count


# Debugging
#if (__name__ == '__main__'):
#    main()




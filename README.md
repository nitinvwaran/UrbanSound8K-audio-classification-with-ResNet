# UrbanSound8K Audio Classification with ResNet-18

This project aims to classify the environmental sounds from the UrbanSound8K dataset, using a ResNet-18 architecture. <br />

**Test Accuracy: TBD** <br />

**Training Accuracy: 100%!**
![alt text](https://github.com/nitinvwaran/UrbanSound8K-audio-classification-with-ResNet/blob/master/misc/accuracy_resnet_18.PNG) <br />

**Validation Accuracy: 77.26%**
![alt text](https://github.com/nitinvwaran/UrbanSound8K-audio-classification-with-ResNet/blob/master/misc/accuracy_resnet_validation.PNG) <br />

**Training Loss Over Time** 
![alt text](https://github.com/nitinvwaran/UrbanSound8K-audio-classification-with-ResNet/blob/master/misc/loss_resnet_18.PNG) <br />

**UrbanSound 8K Dataset** <br />
The UrbanSound8K dataset information can be found here: https://urbansounddataset.weebly.com/urbansound8k.html <br />
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.

**Residual Networks** <br />
The ResNet-18 architecture is the residual network architecture with 18 layers. More information on Residual Networks can be found in the link to the paper:  ('Deep Residual Learning for Image Recognition': https://arxiv.org/abs/1512.03385). <br /> Residual Networks were initially used for Image Object Detection and Image Classification. 

**ResNet-18 on UrbanSound8K** <br />
Inspired by work from Google Research for Audio Classification ('CNN Architectures for Large Scale Audio Classification': https://ai.google/research/pubs/pub45611), this github project was born with the idea to use Residual Networks for Audio Classification of Environmental Sound data such as the UrbanSound8K. The ResNet-18 layer was selected, with the aim to create a smaller model that could be optimized and deployed for smaller devices such as Mobile Phone and Raspberry Pi. 

The original ResNet building block is used (Convolution -> Batch Normalization -> ReLU -> Convolution -> Batch Normalization -> Shortcut Addition -> ReLU), as can be seen modeled in the below diagram <br /> (Source: Kaiming He et. al, 'Identity Mappings in Deep Residual Networks') <br />
![alt text](https://github.com/nitinvwaran/UrbanSound8K-audio-classification-with-ResNet/blob/master/misc/original_resnet_block.PNG)

**Data Pre-Processing** <br />
The following Data pre-processing steps were applied:
1. All .wav files were downsampled to 16KHz with single (Mono) channel
2. Spectogram was extracted from the audio signal, and a Mel Filterbank was applied to the raw spectogram (using the librosa package).
   The number of Mel Filterbanks applied is 128.
3. Log of the Mel Filterbanks was taken, after adding a small offset (1e-10)
4. The number of frames is extracted from each .wav file. Any frames after the 75th frame from the .wav file are discarded. If the .wav file      has less than 75 frames, zero padding is applied. To generate the frames, the default settings from the librosa package were used.
5. Batch-size of 250 was selected, to give the inputs to the model as [250,75,128] ([batch_size, frame_length, number_mel_filterbanks_frequecies]
6. Data is split into train-dev-test split of roughly 60-20-20
<br />

**Model, Optimizer, and Loss details**
1. Resnet-18 Model is used with ResNet v1 block, with the final layer being a dense layer for 10 classes.
2. Adam Optimizer is used with initial learning rate of 0.01.
3. Categorical Cross-Entropy Loss is used with mean reduction.
4. Full-batch-size is used for training and gradient descent, given the small dataset size. 
5. Model is run through 100 epochs.
<br />

















**Acknowledgements**
The tensorflow building blocks for the ResNet-18 architecture were adapted from the following github account: https://github.com/dalgu90/resnet-18-tensorflow. The adaptation is a simpler version of the original residual network building blocks from the github account.



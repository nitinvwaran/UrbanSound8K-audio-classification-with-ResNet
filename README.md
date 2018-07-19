# UrbanSound8K Audio Classification with ResNet-18

This project aims to classify the environmental sounds from the UrbanSound8K dataset, using a ResNet-18 architecture. 

**UrbanSound 8K Dataset** <br />
The UrbanSound8K dataset information can be found here: https://urbansounddataset.weebly.com/urbansound8k.html <br />
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.

**Residual Networks** <br />
The ResNet-18 architecture is the residual network architecture with 18 layers. More information on Residual Networks can be found in the link to the paper:  ('Deep Residual Learning for Image Recognition': https://arxiv.org/abs/1512.03385). <br /> Residual Networks were initially used for Image Object Detection and Image Classification. 

**ResNet-18 on UrbanSound8K** <br />
Inspired by work from Google Research for Audio Classification ('CNN Architectures for Large Scale Audio Classification': https://ai.google/research/pubs/pub45611), this github project was born with the idea to use Residual Networks for Audio Classification of Environmental Sound data such as the UrbanSound8K. The ResNet-18 layer was selected, with the aim to create a smaller model that could be optimized and deployed for smaller devices such as Mobile Phone and Raspberry Pi. 

The original ResNet building block is used (Convolution -> Batch Normalization -> ReLU -> Convolution -> Batch Normalization -> Shortcut Addition -> ReLU), as can be seen modeled in the below diagram:

![alt text](https://github.com/nitinvwaran/UrbanSound8K-audio-classification-with-ResNet/blob/master/misc/original_resnet_block.PNG)



















**Acknowledgements**
The tensorflow building blocks for the ResNet-18 architecture were adapted from the following github account: https://github.com/dalgu90/resnet-18-tensorflow. The adaptation is a simpler version of the original residual network building blocks from the github account.



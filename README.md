# Using-Transfer-Learning-to-classify-new-voice-commands-

This project aims to implement automatic graph-versioning with transfer learning for classifying new voice command files. The new versions correspond to different versions of the final softmax layers, each version with additional new labels to train.

The project is implemented fully in Tensorflow, using single-word voice commands data from the Tensorflow Simple Audio Recognition Tutorial: https://www.tensorflow.org/tutorials/audio_recognition. 

How It Works:

1. Partition your voice command files into train, validation, and test folders. 

2. Edit the labels_meta.txt file, to add the number of labels / voice commands that need to be trained

3. If this is your first time using the project, create a baseline tensorflow graph model, trained on a defined number of labels from labels_meta.txt. 

4. Now that a baseline graph is created, add mode train, validation, and test splits for any new voice command files to be added to the model

5. You can now choose to trigger training from scratch on all the labels (old + new) , or trigger transfer learning only based on the new labels. 
If you choose the latter option, bottleneck files are created for all files in the train and validation folders, and a new graph version is automatically created by re-training only the weights in the final layer (with number of cells = label count). 

6. Use the habits_inference.py module to test the graph with test data.

There are two main benefits from this project:
1. Graph version after transfer learning is automatically created.
2. The main Neural Net Architecture has been modularized in the function build_graph(). This can easily be replaced with another architecture, or any model from the Tensorflow hub. The current model is very simple, based on a single CNN layer with two FC layers following.





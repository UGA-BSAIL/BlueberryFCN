# Fully convolutional networks for blueberry bruising and calyx segmentation using hyperspectral transmittance imaging
Project: Fully convolutional networks for blueberry bruising and calyx segmentation using hyperspectral transmittance imaging

Please download hypwespectral data of blueberry at https://drive.google.com/open?id=16yoBLd2MVOx2ESNGUGZkE4FBO83OpV6v
and unzip it at the code downloding path.

The VGG-16 was used as backbone achitecture. I modified source code to feed in hyperspectral data and can be trained from scratch.
The source code is from https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation.git

“TRAIN_blue.py” is for training the model of 87-newModel.

The logs, training loss, validation loss, checkpoint, and re-trained models will be saved at “logs” folder.

“Evaluate_Net_IOU_blue.py” is for calculating IOU accuracy for the test dataset after training.

The model architecture is at “BuildNetVgg16.py”. 
“Data_Reader.py” is for reading data. 

“Inference_blue.py” is to see the predicted image, the predicted image will be saved at “Output”.   

“Inference_see_feature_map.py” is to see feature map of one blueberry, the blueberry image is at “\blueberry_FCN\programsANDdata\FCN\test_87scratch\f_map”.

“total_parameters.py” is to calculate the numbers of weights in the model. You need to predict or do IOU calculation first to run this program.

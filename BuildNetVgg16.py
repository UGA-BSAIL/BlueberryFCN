################Class which build the fully convolutional neural net###########################################################

import inspect
import os
import TensorflowUtils as utils
import numpy as np
import tensorflow as tf


#VGG_MEAN = [99, 101, 104,108, 112, 116, 119, 121, 123]# Mean value of pixels  in R G and B channels
#VGG_MEAN = [74.88,73.25,72.79,73.36,74.28,76.67,79.16,82.66,86.49,90.50,95.06,99.84,104.35,109.31,113.20,116.52,119.91,122.05,123.82,124.35,124.73,123.34,121.43,119.17,116.17,112.98,109.09,105.81,101.62,97.83,92.77,85.52,75.92,64.63,52.60,41.74,34.55,30.11,27.71,26.16,25.22,24.58,24.01,23.46,23.12,22.68,22.55,22.67,22.80,23.02,23.26,23.56,23.93,24.24,24.59,24.89,25.22,25.45,25.62,25.64,25.52,25.20,24.78,24.10,23.18,22.17,20.89,19.61,18.35,17.05,15.75,14.56,13.57,12.70,12.05,11.61,11.27,10.97,10.75,10.62,10.42,10.28,10.17,10.24,10.17,10.12,10.07]
#========================Class for building the FCN neural network based on VGG16==================================================================================
class BUILD_NET_VGG16:
    def __init__(self, vgg16_npy_path=None):
        # if vgg16_npy_path is None:
        #     path = inspect.getfile(BUILD_NET_VGG16)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg16.npy")
        #     vgg16_npy_path = path
        #
        #     print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item() #Load weights of trained VGG16 for encoder
        print("npy file loaded")
########################################Build Net#####################################################################################################################
    def build(self, rgb,NUM_CLASSES,keep_prob):  #Build the fully convolutional neural network (FCN) and load weight for decoder based on trained VGG16 network
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values 0-255
        """
        self.SumWeights = tf.constant(0.0, name="SumFiltersWeights") #Sum of weights of all filters for weight decay loss


        print("build model started")
        # rgb_scaled = rgb * 255.0

        # Convert RGB to BGR and substract pixels mean
        # red1, red2, red3, red4, red5, red6, red7, red8, red9, red10,red11, red12, red13, red14, red15, red16, red17, red18, red19,red20,red21, red22, red23, red24, red25, red26, red27, red28, red29, red30,red31, red32, red33, red34, red35, red36, red37, red38, red39, red40,red41, red42, red43, red44, red45, red46, red47, red48, red49, red50,red51, red52, red53, red54, red55, red56, red57, red58, red59, red60,red61, red62, red63, red64, red65, red66, red67, red68, red69, red70,red71, red72, red73, red74, red75, red76, red77, red78, red79, red80,red81, red82, red83, red84, red85, red86, red87 = tf.split(axis=3, num_or_size_splits=87, value=rgb)

        # rgb = tf.concat(axis=3, values=[
            # red1 - VGG_MEAN[0],
            # red2 - VGG_MEAN[1],
            # red3 - VGG_MEAN[2],
            # red4 - VGG_MEAN[3],
            # red5 - VGG_MEAN[4],
            # red6 - VGG_MEAN[5],
            # red7 - VGG_MEAN[6],
            # red8 - VGG_MEAN[7],
            # red9 - VGG_MEAN[8],
            # red10 - VGG_MEAN[9],
            # red11 - VGG_MEAN[10],
            # red12 - VGG_MEAN[11],
            # red13 - VGG_MEAN[12],
            # red14 - VGG_MEAN[13],
            # red15 - VGG_MEAN[14],
            # red16 - VGG_MEAN[15],
            # red17 - VGG_MEAN[16],
            # red18 - VGG_MEAN[17],
            # red19 - VGG_MEAN[18],
            # red20 - VGG_MEAN[19],
            # red21 - VGG_MEAN[20],
            # red22 - VGG_MEAN[21],
            # red23 - VGG_MEAN[22],
            # red24 - VGG_MEAN[23],
            # red25 - VGG_MEAN[24],
            # red26 - VGG_MEAN[25],
            # red27 - VGG_MEAN[26],
            # red28 - VGG_MEAN[27],
            # red29 - VGG_MEAN[28],
            # red30 - VGG_MEAN[29],
            # red31 - VGG_MEAN[30],
            # red32 - VGG_MEAN[31],
            # red33 - VGG_MEAN[32],
            # red34 - VGG_MEAN[33],
            # red35 - VGG_MEAN[34],
            # red36 - VGG_MEAN[35],
            # red37 - VGG_MEAN[36],
            # red38 - VGG_MEAN[37],
            # red39 - VGG_MEAN[38],
            # red40 - VGG_MEAN[39],
            # red41 - VGG_MEAN[40],
            # red42 - VGG_MEAN[41],
            # red43 - VGG_MEAN[42],
            # red44 - VGG_MEAN[43],
            # red45 - VGG_MEAN[44],
            # red46 - VGG_MEAN[45],
            # red47 - VGG_MEAN[46],
            # red48 - VGG_MEAN[47],
            # red49 - VGG_MEAN[48],
            # red50 - VGG_MEAN[49],
            # red51 - VGG_MEAN[50],
            # red52 - VGG_MEAN[51],
            # red53 - VGG_MEAN[52],
            # red54 - VGG_MEAN[53],
            # red55 - VGG_MEAN[54],
            # red56 - VGG_MEAN[55],
            # red57 - VGG_MEAN[56],
            # red58 - VGG_MEAN[57],
            # red59 - VGG_MEAN[58],
            # red60 - VGG_MEAN[59],
            # red61 - VGG_MEAN[60],
            # red62 - VGG_MEAN[61],
            # red63 - VGG_MEAN[62],
            # red64 - VGG_MEAN[63],
            # red65 - VGG_MEAN[64],
            # red66 - VGG_MEAN[65],
            # red67 - VGG_MEAN[66],
            # red68 - VGG_MEAN[67],
            # red69 - VGG_MEAN[68],
            # red70 - VGG_MEAN[69],
            # red71 - VGG_MEAN[70],
            # red72 - VGG_MEAN[71],
            # red73 - VGG_MEAN[72],
            # red74 - VGG_MEAN[73],
            # red75 - VGG_MEAN[74],
            # red76 - VGG_MEAN[75],
            # red77 - VGG_MEAN[76],
            # red78 - VGG_MEAN[77],
            # red79 - VGG_MEAN[78],
            # red80 - VGG_MEAN[79],
            # red81 - VGG_MEAN[80],
            # red82 - VGG_MEAN[81],
            # red83 - VGG_MEAN[82],
            # red84 - VGG_MEAN[83],
            # red85 - VGG_MEAN[84],
            # red86 - VGG_MEAN[85],
            # red87 - VGG_MEAN[86],
           
            
        # ])

#-----------------------------Build network encoder based on VGG16 network and load the trained VGG16 weights-----------------------------------------
       #Layer 1
#        kernel = _variable_with_weight_decay('weights',
#                                         shape=[1, 1, 9, 3],
#                                         stddev=5e-2,
#                                         wd=None)
#        self.addconv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
        #layer 1
        self.conv1_1 = tf.layers.conv2d(
      inputs=rgb,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv1_1")
        self.conv1_2 = tf.layers.conv2d(
      inputs=self.conv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv1_2")
        self.pool1 = tf.layers.max_pooling2d(
      inputs=self.conv1_2,
      pool_size = [2 , 2],
      strides = 2,
      padding='same',
      data_format='channels_last',
      name="pool1")
        #layer 2
        self.conv2_1 = tf.layers.conv2d(
      inputs=self.pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv2_1")
        self.conv2_2 = tf.layers.conv2d(
      inputs=self.conv2_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv2_2")
        self.pool2 = tf.layers.max_pooling2d(
      inputs=self.conv2_2,
      pool_size = [2 , 2],
      strides = 2,
      padding='same',
      data_format='channels_last',
      name="pool2")
        #layer 3
        self.conv3_1 = tf.layers.conv2d(
      inputs=self.pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv3_1")
        self.conv3_2 = tf.layers.conv2d(
      inputs=self.conv3_1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv3_2")
        self.conv3_3 = tf.layers.conv2d(
      inputs=self.conv3_2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv3_3")
        self.pool3 = tf.layers.max_pooling2d(
      inputs=self.conv3_3,
      pool_size = [2 , 2],
      strides = 2,
      padding='same',
      data_format='channels_last',
      name="pool3")
        #layer 4
        self.conv4_1 = tf.layers.conv2d(
      inputs=self.pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv4_1")
        self.conv4_2 = tf.layers.conv2d(
      inputs=self.conv4_1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv4_2")
        self.conv4_3 = tf.layers.conv2d(
      inputs=self.conv4_2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv4_3")
        self.pool4 = tf.layers.max_pooling2d(
      inputs=self.conv4_3,
      pool_size = [2 , 2],
      strides = 2,
      padding='same',
      data_format='channels_last',
      name="pool4")
        #layer 5
        self.conv5_1 = tf.layers.conv2d(
      inputs=self.pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv5_1")
        self.conv5_2 = tf.layers.conv2d(
      inputs=self.conv5_1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv5_2")
        self.conv5_3 = tf.layers.conv2d(
      inputs=self.conv5_2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
#      kernel_initializer = tf.initializers.glorot_normal,
      activation=tf.nn.relu,
      name="conv5_3")
        self.pool5 = tf.layers.max_pooling2d(
      inputs=self.conv5_3,
      pool_size = [2 , 2],
      strides = 2,
      padding='same',
      data_format='channels_last',
      name="pool5")
        
       
#        self.conv1_1 = self.conv_layer(self.addconv, "conv1_1") #Build Convolution layer and load weights
#        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")#Build Convolution layer +Relu and load weights
#        self.pool1 = self.max_pool(self.conv1_2, 'pool1') #Max Pooling
       # Layer 2
#        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
#        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
#       self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Layer 3
#        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
#        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
#        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
#        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Layer 4
#        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
#        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
#        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
#        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Layer 5
#        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
#        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
#        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
#        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
##-----------------------Build Net Fully connvolutional layers------------------------------------------------------------------------------------
        W6 = utils.weight_variable([7, 7, 512, 4096],name="W6")  # Create tf weight for the new layer with initial weights with normal random distrubution mean zero and std 0.02
        b6 = utils.bias_variable([4096], name="b6")  # Create tf biase for the new layer with initial weights of 0
        self.conv6 = utils.conv2d_basic(self.pool5 , W6, b6)  # Check the size of this net input is it same as input or is it 1X1
        self.relu6 = tf.nn.relu(self.conv6, name="relu6")
        # if FLAGS.debug: utils.add_activation_summary(relu6)
        self.relu_dropout6 = tf.nn.dropout(self.relu6,keep_prob=keep_prob)  # Apply dropout for traning need to be added only for training

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")  # 1X1 Convloution
        b7 = utils.bias_variable([4096], name="b7")
        self.conv7 = utils.conv2d_basic(self.relu_dropout6, W7, b7)  # 1X1 Convloution
        self.relu7 = tf.nn.relu(self.conv7, name="relu7")
        # if FLAGS.debug: utils.add_activation_summary(relu7)
        self.relu_dropout7 = tf.nn.dropout(self.relu7, keep_prob=keep_prob)  # Another dropout need to be used only for training

        W8 = utils.weight_variable([1, 1, 4096, NUM_CLASSES],name="W8")  # Basically the output num of classes imply the output is already the prediction this is flexible can be change however in multinet class number of 2 give good results
        b8 = utils.bias_variable([NUM_CLASSES], name="b8")
        self.conv8 = utils.conv2d_basic(self.relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
#-------------------------------------Build Decoder --------------------------------------------------------------------------------------------------
        # now to upscale to actual image size
        deconv_shape1 = self.pool4.get_shape()  # Set the output shape for the the transpose convolution output take only the depth since the transpose convolution will have to have the same depth for output
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_CLASSES],name="W_t1")  # Deconvolution/transpose in size 4X4 note that the output shape is of  depth NUM_OF_CLASSES this is not necessary in will need to be fixed if you only have 2 catagories
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        self.conv_t1 = utils.conv2d_transpose_strided(self.conv8, W_t1, b_t1, output_shape=tf.shape(self.pool4))  # Use strided convolution to double layer size (depth is the depth of pool4 for the later element wise addition
        self.fuse_1 = tf.add(self.conv_t1, self.pool4, name="fuse_1")  # Add element wise the pool layer from the decoder

        deconv_shape2 = self.pool3.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        self.conv_t2 = utils.conv2d_transpose_strided(self.fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
        self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")

        shape = tf.shape(rgb)
        W_t3 = utils.weight_variable([16, 16, NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_CLASSES], name="b_t3")

        self.Prob = utils.conv2d_transpose_strided(self.fuse_2, W_t3, b_t3, output_shape=[shape[0], shape[1], shape[2], NUM_CLASSES], stride=8)
     #--------------------Transform  probability vectors to label maps-----------------------------------------------------------------
        self.Pred = tf.argmax(self.Prob, dimension=3, name="Pred")

        print("FCN model built")
#####################################################################################################################################################
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
############################################################################################################################################################
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

############################################################################################################################################################
    def conv_layer_NoRelu(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

#########################################Build fully convolutional layer##############################################################################################################
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
######################################Get VGG filter ############################################################################################################
    def get_conv_filter(self, name):
        var=tf.Variable(self.data_dict[name][0], name="filter_" + name)
        self.SumWeights+=tf.nn.l2_loss(var)
        return var
##################################################################################################################################################
    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_"+name)
#############################################################################################################################################
    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights_"+name)

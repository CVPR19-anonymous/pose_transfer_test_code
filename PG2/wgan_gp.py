import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot
import pdb

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs, MODE):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, MODE, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output, MODE)

    return shortcut + (0.3*output)

###########################################################################################

class WGAN_GP(object):

    def __init__(self, DATA_DIR='', MODE='wgan-gp', DIM=64, BATCH_SIZE=64, ITERS=200000, 
                    LAMBDA=10, G_OUTPUT_DIM=128*64*3, IMG_H=128, IMG_W=64):
        # Download 64x64 ImageNet at http://image-net.org/small/download.php and
        # fill in the path to the extracted files here!
        self.DATA_DIR = DATA_DIR
        # if len(self.DATA_DIR) == 0:
        #     raise Exception('Please specify path to data directory in wgan_gp_64x64.py!')
        self.MODE = MODE # dcgan, wgan, wgan-gp, lsgan
        self.DIM = DIM # self.MODEl self.DIMensionality
        self.BATCH_SIZE = BATCH_SIZE # Batch size. Must be a multiple of self.N_GPUS
        self.ITERS = ITERS # How many iterations to train for
        self.LAMBDA = LAMBDA # Gradient penalty self.LAMBDA hyperparameter
        self.G_OUTPUT_DIM = G_OUTPUT_DIM # Number of pixels in each iamge
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W

        self.CRITIC_ITERS = 5 # How many iterations to train the critic for
        self.N_GPUS = 1 # Number of GPUs
        self.DEVICES = ['/gpu:{}'.format(i) for i in xrange(self.N_GPUS)]

        lib.print_model_settings(locals().copy())

    def GeneratorAndDiscriminator(self):
        """
        Choose which generator and discriminator architecture to use by
        uncommenting one of these lines.
        """

        # Baseline (G: DCGAN, D: DCGAN)
        return self.DCGANGenerator, self.DCGANDiscriminator

        # No BN and constant number of filts in G
        # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

        # 512-dim 4-layer ReLU MLP G
        # return FCGenerator, DCGANDiscriminator

        # No normalization anywhere
        # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

        # Gated multiplicative nonlinearities everywhere
        # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

        # tanh nonlinearities everywhere
        # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
        #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

        # 101-layer ResNet G and D
        # return ResnetGenerator, ResnetDiscriminator

        raise Exception('You must choose an architecture!')

    # ! Generators

    def FCGenerator(self, n_samples, noise=None, FC_DIM=512):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = ReLULayer('Generator.1', 128, FC_DIM, noise)
        output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
        output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
        output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
        output = lib.ops.linear.Linear('Generator.Out', FC_DIM, self.G_OUTPUT_DIM, output)

        output = tf.tanh(output)

        return output

    def DCGANGenerator(self, n_samples, noise=None, dim=64, bn=True, nonlinearity=tf.nn.relu):
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 8*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 8, 4])
        if bn:
            output = Batchnorm('Generator.BN1', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN4', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, self.G_OUTPUT_DIM])


    def AEDCGANGenerator(self, n_samples, noise=None, dim=64, bn=True, nonlinearity=tf.nn.relu):
        output = tf.reshape(inputs, [-1, input_dim, 128, 64])

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        ####
        output = lib.ops.conv2d.Conv2D('AEGAN.Encoder.1', input_dim, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('AEGAN.Encoder.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('AEGAN.Encoder.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('AEGAN.Encoder.3', 2*dim, 4*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('AEGAN.Encoder.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('AEGAN.Encoder.4', 4*dim, 8*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('AEGAN.Encoder.BN4', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = tf.reshape(output, [-1, 8*4*8*dim])
        output = lib.ops.linear.Linear('AEGAN.Encoder.Output', 8*4*8*dim, 128, output)

        ####
        output = lib.ops.linear.Linear('AEGAN.Decoder.Input', 128, 8*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 8, 4])
        if bn:
            output = Batchnorm('AEGAN.Decoder.BN1', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('AEGAN.Decoder.2', 8*dim, 4*dim, 5, output)
        if bn:
            output = Batchnorm('AEGAN.Decoder.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('AEGAN.Decoder.3', 4*dim, 2*dim, 5, output)
        if bn:
            output = Batchnorm('AEGAN.Decoder.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('AEGAN.Decoder.4', 2*dim, dim, 5, output)
        if bn:
            output = Batchnorm('AEGAN.Decoder.BN4', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('AEGAN.Decoder.5', dim, 3, 5, output)
        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, self.G_OUTPUT_DIM])


    def WGANPaper_CrippledDCGANGenerator(self, n_samples, noise=None, dim=64):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 8*4*dim, noise)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, dim, 8, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, self.G_OUTPUT_DIM])

    def ResnetGenerator(self, n_samples, noise=None, dim=64):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 8*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 8, 4])

        for i in xrange(6):
            output = ResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, self.MODE, resample=None)
        output = ResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, self.MODE, resample='up')
        for i in xrange(6):
            output = ResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, self.MODE, resample=None)
        output = ResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, self.MODE, resample='up')
        for i in xrange(6):
            output = ResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, self.MODE, resample=None)
        output = ResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, self.MODE, resample='up')
        for i in xrange(6):
            output = ResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, self.MODE, resample=None)
        output = ResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, self.MODE, resample='up')
        for i in xrange(5):
            output = ResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, self.MODE, resample=None)

        output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
        output = tf.tanh(output / 5.)

        return tf.reshape(output, [-1, self.G_OUTPUT_DIM])


    def MultiplicativeDCGANGenerator(self, n_samples, noise=None, dim=64, bn=True):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 8*4*8*dim*2, noise)
        output = tf.reshape(output, [-1, 8*dim*2, 8, 4])
        if bn:
            output = Batchnorm('Generator.BN1', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
        if bn:
            output = Batchnorm('Generator.BN2', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
        if bn:
            output = Batchnorm('Generator.BN3', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
        if bn:
            output = Batchnorm('Generator.BN4', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, self.G_OUTPUT_DIM])

    # ! Discriminators

    def MultiplicativeDCGANDiscriminator(self, inputs, dim=64, bn=True, name=''):
        output = tf.reshape(inputs, [-1, 3, 128, 64])

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.1', 3, dim*2, 5, output, stride=2)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN2', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN3', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN4', [0,2,3], output, self.MODE)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = tf.reshape(output, [-1, 8*4*8*dim])
        output = lib.ops.linear.Linear(name+'Discriminator.Output', 8*4*8*dim, 1, output)

        return tf.reshape(output, [-1])


    def ResnetDiscriminator(self, inputs, dim=64, name=''):
        output = tf.reshape(inputs, [-1, 3, 128, 64])
        output = lib.ops.conv2d.Conv2D(name+'Discriminator.In', 3, dim/2, 1, output, he_init=False)

        for i in xrange(5):
            output = ResidualBlock(name+'Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, self.MODE, resample=None)
        output = ResidualBlock(name+'Discriminator.Down1', dim/2, dim*1, 3, output, self.MODE, resample='down')
        for i in xrange(6):
            output = ResidualBlock(name+'Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, self.MODE, resample=None)
        output = ResidualBlock(name+'Discriminator.Down2', dim*1, dim*2, 3, output, self.MODE, resample='down')
        for i in xrange(6):
            output = ResidualBlock(name+'Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, self.MODE, resample=None)
        output = ResidualBlock(name+'Discriminator.Down3', dim*2, dim*4, 3, output, self.MODE, resample='down')
        for i in xrange(6):
            output = ResidualBlock(name+'Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, self.MODE, resample=None)
        output = ResidualBlock(name+'Discriminator.Down4', dim*4, dim*8, 3, output, self.MODE, resample='down')
        for i in xrange(6):
            output = ResidualBlock(name+'Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, self.MODE, resample=None)

        output = tf.reshape(output, [-1, 8*4*8*dim])
        output = lib.ops.linear.Linear(name+'Discriminator.Output', 8*4*8*dim, 1, output)

        return tf.reshape(output / 5., [-1])


    def FCDiscriminator(self, inputs, FC_DIM=512, n_layers=3, reuse=False, name=''):
        output = LeakyReLULayer(name+'Discriminator.Input', self.G_OUTPUT_DIM, FC_DIM, inputs)
        for i in xrange(n_layers):
            output = LeakyReLULayer(name+'Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
        output = lib.ops.linear.Linear(name+'Discriminator.Out', FC_DIM, 1, output)

        return tf.reshape(output, [-1])

    def DCGANDiscriminator(self, inputs, input_dim=3, dim=64, bn=True, nonlinearity=LeakyReLU, name=''):
        # output = tf.reshape(inputs, [-1, input_dim, 128, 64])
        output = inputs

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.1', input_dim, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN4', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = tf.reshape(output, [-1, 8*4*8*dim])
        output = lib.ops.linear.Linear(name+'Discriminator.Output', 8*4*8*dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])

    def DCGANDiscriminatorAttr(self, inputs, attr_num, input_dim=3, dim=64, keep_prob=1, bn=True, nonlinearity=LeakyReLU, name=''):
        # output = tf.reshape(inputs, [-1, input_dim, 8, 4])
        # pdb.set_trace()
        output = inputs

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.1', input_dim, dim, 5, output, stride=2)
        output = nonlinearity(output)
        output = tf.nn.dropout(output, keep_prob)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)
        output = tf.nn.dropout(output, keep_prob)

        output = tf.reshape(name+output, [-1, 2*1*2*dim])
        output = lib.ops.linear.Linear(name+'Discriminator.Output1', 2*1*2*dim, 512, output)
        output = nonlinearity(output)
        output = tf.nn.dropout(output, keep_prob)
        output = lib.ops.linear.Linear(name+'Discriminator.Output2', 512, attr_num, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])

    def DCGANDiscriminator_256(self, inputs, input_dim=3, dim=64, bn=True, nonlinearity=LeakyReLU, name=''):
        # output = tf.reshape(inputs, [-1, input_dim, 256, 256])
        output = inputs
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.1', input_dim, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN4', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.5', 8*dim, 8*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN5', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = tf.reshape(output, [-1, 8*8*8*dim])
        output = lib.ops.linear.Linear(name+'Discriminator.Output', 8*8*8*dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])


    def DCGANDiscriminatorRegion(self, inputs, input_dim=3, dim=64, bn=True, nonlinearity=LeakyReLU, name=''):
        # output = tf.reshape(inputs, [-1, input_dim, 128, 64])
        output = inputs
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.1', input_dim, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN2', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        if bn:
            output = Batchnorm(name+'Discriminator.BN3', [0,2,3], output, self.MODE)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(name+'Discriminator.4', 4*dim, 1, 5, output, stride=1)
        # if bn:
        #     output = Batchnorm(name+'Discriminator.BN4', [0,2,3], output, self.MODE)
        # output = nonlinearity(output)

        # output = tf.reshape(output, [-1, 8*4*8*dim])
        # output = lib.ops.linear.Linear(name+'Discriminator.Output', 8*4*8*dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, 16, 8])

    ## ref code: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py#L400
    ## receptive_field_sizes:https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    # def PatchDiscriminator(self, inputs, input_dim=3, dim=64, bn=True, n_layers=3, nonlinearity=LeakyReLU, name=''):
    def PatchDiscriminator(self, inputs, input_dim=3, dim=64, bn=True, n_layers=2, nonlinearity=LeakyReLU, name=''):
        # output = tf.reshape(inputs, [-1, input_dim, 128, 64])
        output = inputs

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D_reflect(name+'Discriminator.1', input_dim, dim, 4, output, stride=2)
        output = nonlinearity(output)

        for i in range(n_layers):
            in_channels = output.get_shape().as_list()[1]
            out_channels = dim * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            output = lib.ops.conv2d.Conv2D_reflect(name+'Discriminator.%d'%(i+2), in_channels, out_channels, 4, output, stride=stride)
            if bn:
                output = Batchnorm(name+'Discriminator.BN%d'%(i+2), [0,2,3], output, self.MODE)
            output = nonlinearity(output)

        in_channels = output.get_shape().as_list()[1]
        output = lib.ops.conv2d.Conv2D_reflect(name+'Discriminator.%d'%(n_layers+2), in_channels, 1, 4, output, stride=1)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return output



    def main(self):
        Generator, Discriminator = self.GeneratorAndDiscriminator()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

            all_real_data_conv = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, 3, 64, 64])
            if tf.__version__.startswith('1.'):
                split_real_data_conv = tf.split(all_real_data_conv, len(self.G_OUTPUT_DIM))
            else:
                split_real_data_conv = tf.split(0, len(self.G_OUTPUT_DIM), all_real_data_conv)
            gen_costs, disc_costs = [],[]

            for device_index, (device, real_data_conv) in enumerate(zip(self.G_OUTPUT_DIM, split_real_data_conv)):
                with tf.device(device):

                    real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [self.BATCH_SIZE/len(self.G_OUTPUT_DIM), self.G_OUTPUT_DIM])
                    fake_data = Generator(self.BATCH_SIZE/len(self.G_OUTPUT_DIM))

                    disc_real = Discriminator(real_data)
                    disc_fake = Discriminator(fake_data)

                    if self.MODE == 'wgan':
                        gen_cost = -tf.reduce_mean(disc_fake)
                        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    elif self.MODE == 'wgan-gp':
                        gen_cost = -tf.reduce_mean(disc_fake)
                        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                        alpha = tf.random_uniform(
                            shape=[self.BATCH_SIZE/len(self.G_OUTPUT_DIM),1], 
                            minval=0.,
                            maxval=1.
                        )
                        differences = fake_data - real_data
                        interpolates = real_data + (alpha*differences)
                        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                        disc_cost += self.LAMBDA*gradient_penalty

                    elif self.MODE == 'dcgan':
                        try: # tf pre-1.0 (bottom) vs 1.0 (top)
                            gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                              labels=tf.ones_like(disc_fake)))
                            disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                                labels=tf.zeros_like(disc_fake)))
                            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                                labels=tf.ones_like(disc_real)))                    
                        except Exception as e:
                            gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                            disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))                    
                        disc_cost /= 2.

                    elif self.MODE == 'lsgan':
                        gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                        disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

                    else:
                        raise Exception()

                    gen_costs.append(gen_cost)
                    disc_costs.append(disc_cost)

            gen_cost = tf.add_n(gen_costs) / len(self.G_OUTPUT_DIM)
            disc_cost = tf.add_n(disc_costs) / len(self.G_OUTPUT_DIM)

            if self.MODE == 'wgan':
                gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                                     var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
                disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                     var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

                clip_ops = []
                for var in lib.params_with_name('Discriminator'):
                    clip_bounds = [-.01, .01]
                    clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
                clip_disc_weights = tf.group(*clip_ops)

            elif self.MODE == 'wgan-gp':
                gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                  var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
                disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                   var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

            elif self.MODE == 'dcgan':
                gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                  var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
                disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                   var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

            elif self.MODE == 'lsgan':
                gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                                     var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
                disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                      var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

            else:
                raise Exception()

            # For generating samples
            fixed_noise = tf.constant(np.random.normal(size=(self.BATCH_SIZE, 128)).astype('float32'))
            all_fixed_noise_samples = []
            for device_index, device in enumerate(self.G_OUTPUT_DIM):
                n_samples = self.BATCH_SIZE / len(self.G_OUTPUT_DIM)
                all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
            if tf.__version__.startswith('1.'):
                all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
            else:
                all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
            def generate_image(iteration):
                samples = session.run(all_fixed_noise_samples)
                samples = ((samples+1.)*(255.99/2)).astype('int32')
                lib.save_images.save_images(samples.reshape((self.BATCH_SIZE, 3, 64, 64)), 'samples_{}.png'.format(iteration))


            # Dataset iterator
            train_gen, dev_gen = lib.small_imagenet.load(self.BATCH_SIZE, data_dir=self.DATA_DIR)

            def inf_train_gen():
                while True:
                    for (images,) in train_gen():
                        yield images

            # Save a batch of ground-truth samples
            _x = inf_train_gen().next()
            _x_r = session.run(real_data, feed_dict={real_data_conv: _x})
            _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
            lib.save_images.save_images(_x_r.reshape((self.BATCH_SIZE, 3, 64, 64)), 'samples_groundtruth.png')


            # Train loop
            session.run(tf.initialize_all_variables())
            gen = inf_train_gen()
            for iteration in xrange(self.ITERS):

                start_time = time.time()

                # Train generator
                if iteration > 0:
                    _ = session.run(gen_train_op)

                # Train critic
                if (self.MODE == 'dcgan') or (self.MODE == 'lsgan'):
                    disc_iters = 1
                else:
                    disc_iters = self.CRITIC_ITERS
                for i in xrange(disc_iters):
                    _data = gen.next()
                    _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})
                    if self.MODE == 'wgan':
                        _ = session.run([clip_disc_weights])

                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('time', time.time() - start_time)

                if iteration % 200 == 199:
                    t = time.time()
                    dev_disc_costs = []
                    for (images,) in dev_gen():
                        _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: _data}) 
                        dev_disc_costs.append(_dev_disc_cost)
                    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

                    generate_image(iteration)

                if (iteration < 5) or (iteration % 200 == 199):
                    lib.plot.flush()

                lib.plot.tick()



import tensorflow as tf
import numpy as np
from wavetf import WaveTFFactory
from tensorflow.keras import Input, initializers, regularizers, constraints
from tensorflow.keras import backend as K
#from tensorflow.keras.engine import Layer, InputSpec
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, Flatten, Lambda, Masking, Layer, InputSpec
from tensorflow.keras.activations import tanh, sigmoid
from WaveletUnit import WaveletUnitLayer


wavelet = WaveTFFactory.build('db2', dim=2)
wavelet_i = WaveTFFactory.build('db2', dim=2, inverse=True)


#https://github.com/titu1994/Keras-Group-Normalization
class GroupNormalization(Layer):
    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)
        # standardlization
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

#https://stackoverflow.com/questions/66305623/group-normalization-and-weight-standardization-in-keras
class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)

    def standardize_weight(self, epsilon=1e-5):
        kernel_mean = tf.math.reduce_mean(self.kernel, axis=[0, 1, 2, 3], keepdims=True, name="kernel_mean")
        ker = self.kernel - kernel_mean
        kernel_std = tf.keras.backend.std(ker, axis=[0, 1, 2, 3], keepdims=True)
        ker = ker / (kernel_std + epsilon)
        return ker

    def call(self, inputs, eps=1e-5):
        self.kernel.assign(self.standardize_weight(epsilon=eps))
        return super().call(inputs)

class WSConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        super(WSConv2DTranspose, self).__init__(*args, **kwargs)

    def standardize_weight(self, epsilon=1e-5):
        kernel_mean = tf.math.reduce_mean(self.kernel, axis=[0, 1, 2, 3], keepdims=True, name="kernel_mean")
        ker = self.kernel - kernel_mean
        kernel_std = tf.keras.backend.std(ker, axis=[0, 1, 2, 3], keepdims=True)
        ker = ker / (kernel_std + epsilon)
        return ker

    def call(self, inputs, eps=1e-5):
        self.kernel.assign(self.standardize_weight(epsilon=eps))
        return super().call(inputs)


def batch_norm(map):
    m_h, v_h = tf.nn.moments(map, [0])
    h_batch_norm = tf.nn.batch_normalization(map, mean=m_h, variance=v_h,
                                             offset=tf.constant(0.),
                                             scale=tf.constant(1.),
                                             variance_epsilon=1e-8)
    return h_batch_norm


def Add_and_Norm(feature1, feature2):
    temp = tf.math.add(feature1, feature2)
    temp = BatchNormalization()(temp)
    temp = LeakyReLU()(temp)
    return temp


def Add_and_GNorm(feature1, feature2):
    temp = tf.math.add(feature1, feature2)
    temp = GroupNormalization()(temp)
    temp = LeakyReLU()(temp)
    return temp


def fft_on_axis(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x_fft = tf.signal.rfft2d(x)
    result = tf.transpose(x_fft, perm=[0, 2, 3, 1])
    return result

def ifft_on_axis(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x_ifft = tf.signal.irfft2d(x)
    result = tf.transpose(x_ifft, perm=[0, 2, 3, 1])
    return result

@tf.function
def FourierUnit(feature):
    y_r, y_i = tf.split(feature, num_or_size_splits=2, axis=-1)#fixing

    y_r_nor = tf.nn.conv2d(y_r, tf.random.normal([1, 1, tf.shape(y_r)[-1], tf.shape(y_r)[-1]]), strides=1,
                           padding="SAME")
    y_r_nor = batch_norm(y_r_nor)
    y_r_nor = tf.nn.leaky_relu(y_r_nor)

    y_i_nor = tf.nn.conv2d(y_i, tf.random.normal([1, 1, tf.shape(y_i)[-1], tf.shape(y_i)[-1]]), strides=1,
                           padding="SAME")
    y_i_nor = batch_norm(y_i_nor)
    y_i_nor = tf.nn.leaky_relu(y_i_nor)

    # y = Conv2D(tf.shape(y)[3], (1, 1))(y)
    # f = tf.Variable(tf.random.normal([1, 1, 4002, 4000]))
    #print(y_i.shape)
    y_r3x3 = tf.nn.conv2d(y_r_nor, tf.random.normal([3, 3, tf.shape(y_r)[-1], tf.shape(y_r)[-1]]), strides=1,
                          padding="SAME")

    y_i3x3 = tf.nn.conv2d(y_i_nor, tf.random.normal([3, 3, tf.shape(y_i)[-1], tf.shape(y_i)[-1]]), strides=1,
                          padding="SAME")

    freq_y_i = fft_on_axis(y_i_nor)
    y_rs = tf.math.real(freq_y_i)
    y_is = tf.math.imag(freq_y_i)
    y_s = tf.concat((y_rs, y_is), axis=-1)
    y_s = tf.nn.conv2d(y_s, tf.random.normal([1, 1, tf.shape(y_s)[-1], tf.shape(y_s)[-1]]), strides=1, padding="SAME")
    y_s = batch_norm(y_s)
    y_s = tf.nn.leaky_relu(y_s)

    z_r, z_i = tf.split(y_s, num_or_size_splits=2, axis=-1)
    z = tf.dtypes.complex(z_r, z_i)
    z = ifft_on_axis(z)
    spectral = tf.nn.conv2d(z, tf.random.normal([1, 1, tf.shape(z)[-1], tf.shape(z)[-1]]), strides=1, padding="SAME")

    global_feature = tf.add(y_r3x3, spectral)
    global_feature = batch_norm(global_feature)
    global_feature = tf.nn.leaky_relu(global_feature)

    local_feature = tf.add(y_r3x3, y_i3x3)
    local_feature = batch_norm(local_feature)
    local_feature = tf.nn.leaky_relu(local_feature)

    final = tf.concat((global_feature, local_feature), axis=-1)

    return final

def Residual_block(inputs, channel, filter, strides=(2, 2), dilation=1, padding="same"):

    conv0_1 = Conv2D(channel, (1, 1), strides=(1, 1), padding=padding)(inputs)
    bn0_1 = BatchNormalization()(conv0_1)
    relu0_1 = LeakyReLU()(bn0_1)

    conv0_2 = Conv2D(channel, filter, strides=(1, 1), padding=padding, dilation_rate=dilation)(relu0_1)
    bn0_2 = BatchNormalization()(conv0_2)
    relu0_2 = LeakyReLU()(bn0_2)

    try:
        feature = tf.math.add(relu0_2, inputs)
    except:
        feature = tf.math.add(relu0_2, conv0_1)
    output = Conv2D(channel, (1, 1), strides=strides, padding=padding)(feature)

    return output


def Residual_Gblock(inputs, channel, filter, strides=(2, 2), dilation=1, padding="same"):

    conv0_1 = WSConv2D(channel, (1, 1), strides=(1, 1), padding=padding)(inputs)
    bn0_1 = GroupNormalization()(conv0_1)
    relu0_1 = LeakyReLU()(bn0_1)

    conv0_2 = WSConv2D(channel, filter, strides=(1, 1), padding=padding, dilation_rate=dilation)(relu0_1)
    bn0_2 = GroupNormalization()(conv0_2)
    relu0_2 = LeakyReLU()(bn0_2)

    try:
        feature = tf.math.add(relu0_2, inputs)
    except:
        feature = tf.math.add(relu0_2, conv0_1)
    output = WSConv2D(channel, (1, 1), strides=strides, padding=padding)(feature)
    output = LeakyReLU()(GroupNormalization()(output))

    return output


def slice(feature_map, slice_shape=8, slice_count=16):
    temp = tf.zeros([feature_map.shape[0], slice_shape, slice_shape, 1])
    for i in range(slice_count):
        temp_x = tf.slice(feature_map, [0, slice_shape*i, 0, 0], [feature_map.shape[0], slice_shape*(i+1), feature_map.shape[-2], feature_map.shape[-1]])
        for j in range(slice_count):
            temp_y = tf.slice(tempx, [0, 0, slice_shape*i, 0], [temp_x.shape[0], temp_x.shape[0], slice_shape*(i+1), temp_x.shape[-1]])
            temp = tf.concat((temp, temp_y), axis=-1)

    return temp[1:]


def FQencoder(inputs, feature_size=4096):
    inputs = Input(inputs, dtype="float32")
    res1_1 = Residual_block(inputs, 64, (3, 3), strides=(2, 2), padding="same")
    res2_1 = Residual_block(res1_1, 128, (3, 3), strides=(2, 2), padding="same")
    res3_1 = Residual_block(res2_1, 256, (3, 3), strides=(2, 2), padding="same")
    res4_1 = Residual_block(res3_1, 512, (3, 3), strides=(2, 2), padding="same")
    res4_2 = Residual_block(res4_1, 512, (3, 3), strides=(2, 2), padding="same")
    conv5 = Conv2D(feature_size, (3, 3), strides=(2, 2), padding="same")(res4_2)
    bn5 = LeakyReLU()(BatchNormalization()(conv5))

    encode = bn5
    fu1 = Lambda(FourierUnit)(encode)
    fu2 = Lambda(FourierUnit)(fu1)

    deconv_input4_1 = tf.math.add(conv5, fu2)
    deconv4_1 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_1")(deconv_input4_1)
    debn4_1 = LeakyReLU()(BatchNormalization(name="debn4_1")(deconv4_1))

    deconv_input4_2 = tf.math.add(res4_2, debn4_1)
    deconv4_2 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_2")(deconv_input4_2)
    debn4_2 = LeakyReLU()(BatchNormalization(name="debn4_2")(deconv4_2))

    deconv_input3_1 = tf.math.add(res4_1, debn4_2)
    deconv3_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2),
                                padding="same", name="deconv3_1")(deconv_input3_1)
    debn3_1 = LeakyReLU()(BatchNormalization(name="debn3_1")(deconv3_1))

    deconv_input2_1 = tf.math.add(res3_1, debn3_1)
    deconv2_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2),
                                padding="same", name="deconv2_1")(deconv_input2_1)
    debn2_1 = LeakyReLU()(BatchNormalization(name="debn2_1")(deconv2_1))

    deconv_input1_1 = tf.math.add(res2_1, debn2_1)
    deconv1_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2),
                                padding="same", name="deconv1_1")(deconv_input1_1)
    debn1_1 = LeakyReLU()(BatchNormalization(name="debn1_1")(deconv1_1))

    deconv_input_f = tf.math.add(res1_1, debn1_1)
    deconvf = Conv2DTranspose(3, (3, 3), strides=(2, 2),
                              padding="same", name="deconvf")(deconv_input_f)
    debnf = LeakyReLU()(BatchNormalization(name="debnf")(deconvf))

    recon = tanh(debnf)
    return inputs, recon


def WLencoder(inputs, feature_size=4096):
    inputs = Input(inputs, dtype="float32")
    res1_1 = Residual_block(inputs, 64, (3, 3), strides=(2, 2), padding="same")
    res2_1 = Residual_block(res1_1, 128, (3, 3), strides=(2, 2), padding="same")
    res3_1 = Residual_block(res2_1, 256, (3, 3), strides=(2, 2), padding="same")
    res4_1 = Residual_block(res3_1, 512, (3, 3), strides=(2, 2), padding="same")
    res4_2 = Residual_block(res4_1, 512, (3, 3), strides=(2, 2), padding="same")
    conv5 = Conv2D(feature_size, (3, 3), strides=(2, 2), padding="same")(res4_2)
    bn5 = LeakyReLU()(BatchNormalization()(conv5))

    encode = bn5
    fu1 = Lambda(WaveletUnit)(encode)
    fu2 = Lambda(WaveletUnit)(fu1)

    deconv_input4_1 = tf.math.add(conv5, fu2)
    deconv4_1 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_1")(deconv_input4_1)
    debn4_1 = LeakyReLU()(BatchNormalization(name="debn4_1")(deconv4_1))

    deconv_input4_2 = tf.math.add(res4_2, debn4_1)
    deconv4_2 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_2")(deconv_input4_2)
    debn4_2 = LeakyReLU()(BatchNormalization(name="debn4_2")(deconv4_2))

    deconv_input3_1 = tf.math.add(res4_1, debn4_2)
    deconv3_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2),
                                padding="same", name="deconv3_1")(deconv_input3_1)
    debn3_1 = LeakyReLU()(BatchNormalization(name="debn3_1")(deconv3_1))

    deconv_input2_1 = tf.math.add(res3_1, debn3_1)
    deconv2_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2),
                                padding="same", name="deconv2_1")(deconv_input2_1)
    debn2_1 = LeakyReLU()(BatchNormalization(name="debn2_1")(deconv2_1))

    deconv_input1_1 = tf.math.add(res2_1, debn2_1)
    deconv1_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2),
                                padding="same", name="deconv1_1")(deconv_input1_1)
    debn1_1 = LeakyReLU()(BatchNormalization(name="debn1_1")(deconv1_1))

    deconv_input_f = tf.math.add(res1_1, debn1_1)
    deconvf = Conv2DTranspose(3, (3, 3), strides=(2, 2),
                              padding="same", name="deconvf")(deconv_input_f)
    debnf = LeakyReLU()(BatchNormalization(name="debnf")(deconvf))

    recon = tanh(debnf)
    return inputs, recon


def WLencoder_test(inputs, feature_size=4096):
    inputs = Input(inputs, dtype="float32")
    mask_1 = Masking(mask_value=0.0, input_shape=(256, 256, 3))(inputs)
    res1_1 = Residual_block(mask_1, 64, (3, 3), strides=(2, 2), padding="same")
    #res1_2 = Residual_block(res1_1, 64, (3, 3), strides=(1, 1), padding="same")
    mask_2 = Masking(mask_value=0.0, input_shape=(128, 128, 64))(res1_1)
    res2_1 = Residual_block(mask_2, 128, (3, 3), strides=(2, 2), padding="same")
    #res2_2 = Residual_block(res2_1, 128, (3, 3), strides=(1, 1), padding="same")
    mask_3 = Masking(mask_value=0.0, input_shape=(64, 64, 128))(res2_1)
    res3_1 = Residual_block(mask_3, 256, (3, 3), strides=(2, 2), padding="same")
    conv3 = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(res3_1)
    bn3 = LeakyReLU()(BatchNormalization()(conv3))
    #res3_3 = Residual_block(res3_2, 256, (3, 3), strides=(1, 1), padding="same")
    mask_4 = Masking(mask_value=0.0, input_shape=(32, 32, 256))(bn3)
    res4_1 = Residual_block(mask_4, 512, (3, 3), strides=(2, 2), padding="same")
    conv4 = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(res4_1)
    bn4 = LeakyReLU()(BatchNormalization()(conv4))
    #res4_3 = Residual_block(res4_2, 512, (3, 3), strides=(1, 1), padding="same")
    mask_5 = Masking(mask_value=0.0, input_shape=(16, 16, 512))(bn4)
    res5_1 = Residual_block(mask_5, 512, (3, 3), strides=(2, 2), padding="same")
    conv5 = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(res5_1)
    bn5 = LeakyReLU()(BatchNormalization()(conv5))
    #res5_3 = Residual_block(res5_2, 512, (3, 3), strides=(1, 1), padding="same")
    mask_6 = Masking(mask_value=0.0, input_shape=(8, 8, 512))(bn5)
    conv6 = Conv2D(feature_size, (3, 3), strides=(2, 2), padding="same")(mask_6)
    bn6 = LeakyReLU()(BatchNormalization()(conv6))

    encode = bn6
    mask_7 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(encode)
    fu1 = Lambda(WaveletUnit)(mask_7)
    mask_8 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(fu1)
    fu2 = Lambda(WaveletUnit)(mask_8)

    deconv_input4_1 = tf.math.add(bn6, fu2)
    deconv4_1 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_1")(deconv_input4_1)
    debn4_1 = LeakyReLU()(BatchNormalization(name="debn4_1")(deconv4_1))

    deconv_input4_2 = tf.math.add(mask_6, debn4_1)
    deconv4_2 = Conv2DTranspose(512, (3, 3), strides=(2, 2),
                                padding="same", name="deconv4_2")(deconv_input4_2)
    debn4_2 = LeakyReLU()(BatchNormalization(name="debn4_2")(deconv4_2))

    deconv_input3_1 = tf.math.add(mask_5, debn4_2)
    deconv3_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2),
                                padding="same", name="deconv3_1")(deconv_input3_1)
    debn3_1 = LeakyReLU()(BatchNormalization(name="debn3_1")(deconv3_1))

    deconv_input2_1 = tf.math.add(mask_4, debn3_1)
    deconv2_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2),
                                padding="same", name="deconv2_1")(deconv_input2_1)
    debn2_1 = LeakyReLU()(BatchNormalization(name="debn2_1")(deconv2_1))

    deconv_input1_1 = tf.math.add(mask_3, debn2_1)
    deconv1_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2),
                                padding="same", name="deconv1_1")(deconv_input1_1)
    debn1_1 = LeakyReLU()(BatchNormalization(name="debn1_1")(deconv1_1))

    deconv_input_f = tf.math.add(mask_2, debn1_1)
    deconvf = Conv2DTranspose(3, (3, 3), strides=(2, 2),
                              padding="same", name="deconvf")(deconv_input_f)
    debnf = LeakyReLU()(BatchNormalization(name="debnf")(deconvf))

    recon = tanh(debnf)
    return inputs, recon

def WLencoder_test2(inputs, feature_size=4096):
    inputs = Input(inputs, dtype="float32")
    mask1_1 = Masking(mask_value=0.0, input_shape=(256, 256, 3))(inputs)
    res1_1 = Residual_block(mask1_1, 8, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask1_2 = Masking(mask_value=0.0, input_shape=(128, 128, 8))(res1_1)
    res1_2 = Residual_block(mask1_2, 8, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask2_1 = Masking(mask_value=0.0, input_shape=(128, 128, 8))(res1_2)
    res2_1 = Residual_block(mask2_1, 16, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask2_2 = Masking(mask_value=0.0, input_shape=(64, 64, 16))(res2_1)
    res2_2 = Residual_block(mask2_2, 16, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask3_1 = Masking(mask_value=0.0, input_shape=(64, 64, 16))(res2_2)
    res3_1 = Residual_block(mask3_1, 32, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask3_2 = Masking(mask_value=0.0, input_shape=(32, 32, 32))(res3_1)
    res3_2 = Residual_block(mask3_2, 32, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask3_3 = Masking(mask_value=0.0, input_shape=(32, 32, 32))(res3_2)
    res3_3 = Residual_block(mask3_3, 32, (3, 3), strides=(1, 1), dilation=4, padding="same")

    mask4_1 = Masking(mask_value=0.0, input_shape=(32, 32, 32))(res3_3)
    res4_1 = Residual_block(mask4_1, 64, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask4_2 = Masking(mask_value=0.0, input_shape=(16, 16, 64))(res4_1)
    res4_2 = Residual_block(mask4_2, 64, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask4_3 = Masking(mask_value=0.0, input_shape=(16, 16, 64))(res4_2)
    res4_3 = Residual_block(mask4_3, 64, (3, 3), strides=(1, 1), dilation=4, padding="same")

    mask5_1 = Masking(mask_value=0.0, input_shape=(16, 16, 64))(res4_3)
    res5_1 = Residual_block(mask5_1, 128, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask5_2 = Masking(mask_value=0.0, input_shape=(8, 8, 128))(res5_1)
    res5_2 = Residual_block(mask5_2, 128, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask5_3 = Masking(mask_value=0.0, input_shape=(8, 8, 128))(res5_2)
    res5_3 = Residual_block(mask5_3, 128, (3, 3), strides=(1, 1), dilation=4, padding="same")
    mask5_4 = Masking(mask_value=0.0, input_shape=(8, 8, 128))(res5_3)
    res5_4 = Residual_block(mask5_4, 128, (3, 3), strides=(1, 1), dilation=8, padding="same")

    mask6_1 = Masking(mask_value=0.0, input_shape=(8, 8, 128))(res5_4)
    res6_1 = Residual_block(mask6_1, 256, (3, 3), strides=(1, 1), dilation=1, padding="same")
    mask6_2 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res6_1)
    res6_2 = Residual_block(mask6_2, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask6_3 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res6_2)
    res6_3 = Residual_block(mask6_3, 256, (3, 3), strides=(1, 1), dilation=4, padding="same")
    mask6_4 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res6_3)
    res6_4 = Residual_block(mask6_4, 256, (3, 3), strides=(1, 1), dilation=8, padding="same")

    mask6_5 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res6_4)
    conv6 = Conv2D(feature_size, (3, 3), strides=(1, 1), padding="same")(mask6_5)
    bn6 = LeakyReLU()(BatchNormalization()(conv6))
    encode = bn6

    mask7 = Masking(mask_value=0.0, input_shape=(8, 8, feature_size))(encode)
    fu1 = Lambda(WaveletUnit)(mask7)
    mask8 = Masking(mask_value=0.0, input_shape=(8, 8, feature_size))(fu1)
    fu2 = Lambda(WaveletUnit)(mask8)
    add1 = tf.math.add(fu2, mask7)
    addbn1 = LeakyReLU()(BatchNormalization(name="addbn1")(add1))

    mask9 = Masking(mask_value=0.0, input_shape=(8, 8, feature_size))(addbn1)
    fu3 = Lambda(WaveletUnit)(mask9)
    mask10 = Masking(mask_value=0.0, input_shape=(8, 8, feature_size))(fu3)
    fu4 = Lambda(WaveletUnit)(mask10)
    add2 = tf.math.add(fu4, mask9)
    addbn2 = LeakyReLU()(BatchNormalization(name="addbn2")(add2))

    #decode_input1 = addbn2
    deconv1 = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding="same")(addbn2)
    debn1 = LeakyReLU()(BatchNormalization(name="debn1")(deconv1))

    #decode_input2_1 = tf.math.add(debn1, mask6_5)
    deconv2_1 = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding="same")(debn1)
    debn2_1 = LeakyReLU()(BatchNormalization(name="debn2_1")(deconv2_1))

    #decode_input2_2 = tf.math.add(debn2_1, mask6_1)
    deconv2_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(debn2_1)
    debn2_2 = LeakyReLU()(BatchNormalization(name="debn2_2")(deconv2_2))

    #decode_input3_1 = tf.math.add(debn2_2, mask5_1)
    deconv3_1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(debn2_2)
    debn3_1 = LeakyReLU()(BatchNormalization(name="debn3_1")(deconv3_1))

    #decode_input3_2 = tf.math.add(debn3_1, mask4_1)
    deconv3_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(debn3_1)
    debn3_2 = LeakyReLU()(BatchNormalization(name="debn3_2")(deconv3_2))

    #decode_input4 = tf.math.add(debn3_2, mask3_1)
    #deconv4 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same")(decode_input4)
    #debn4 = LeakyReLU()(BatchNormalization(name="debn4")(deconv4))

    #decode_input5 = tf.math.add(debn3_2, mask3_1)
    deconv5 = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(debn3_2)
    debnf = LeakyReLU()(BatchNormalization(name="debn5")(deconv5))

    recon = tanh(debnf)
    return inputs, recon

def WLencoder_test3(inputs, feature_size=2048):
    inputs = Input(inputs, dtype="float32")
    mask1_1 = Masking(mask_value=0.0, input_shape=(256, 256, 3))(inputs)
    res1_1 = Residual_block(mask1_1, 16, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask1_2 = Masking(mask_value=0.0, input_shape=(128, 128, 16))(res1_1)
    res1_2 = Residual_block(mask1_2, 16, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask2_1 = Masking(mask_value=0.0, input_shape=(128, 128, 16))(res1_2)
    res2_1 = Residual_block(mask2_1, 32, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask2_2 = Masking(mask_value=0.0, input_shape=(64, 64, 32))(res2_1)
    res2_2 = Residual_block(mask2_2, 32, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask3_1 = Masking(mask_value=0.0, input_shape=(64, 64, 32))(res2_2)
    res3_1 = Residual_block(mask3_1, 64, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask3_2 = Masking(mask_value=0.0, input_shape=(32, 32, 64))(res3_1)
    res3_2 = Residual_block(mask3_2, 64, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask3_3 = Masking(mask_value=0.0, input_shape=(32, 32, 64))(res3_2)
    res3_3 = Residual_block(mask3_3, 64, (3, 3), strides=(1, 1), dilation=3, padding="same")

    mask4_1 = Masking(mask_value=0.0, input_shape=(32, 32, 64))(res3_3)
    res4_1 = Residual_block(mask4_1, 128, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask4_2 = Masking(mask_value=0.0, input_shape=(16, 16, 128))(res4_1)
    res4_2 = Residual_block(mask4_2, 128, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask4_3 = Masking(mask_value=0.0, input_shape=(16, 16, 128))(res4_2)
    res4_3 = Residual_block(mask4_3, 128, (3, 3), strides=(1, 1), dilation=3, padding="same")

    mask5_1 = Masking(mask_value=0.0, input_shape=(16, 16, 128))(res4_3)
    res5_1 = Residual_block(mask5_1, 256, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask5_2 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res5_1)
    res5_2 = Residual_block(mask5_2, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask5_3 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res5_2)
    res5_3 = Residual_block(mask5_3, 256, (3, 3), strides=(1, 1), dilation=3, padding="same")
    mask5_4 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res5_3)
    res5_4 = Residual_block(mask5_4, 256, (3, 3), strides=(1, 1), dilation=4, padding="same")

    mask6_1 = Masking(mask_value=0.0, input_shape=(8, 8, 256))(res5_4)
    res6_1 = Residual_block(mask6_1, 512, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask6_2 = Masking(mask_value=0.0, input_shape=(4, 4, 512))(res6_1)
    res6_2 = Residual_block(mask6_2, 512, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask6_3 = Masking(mask_value=0.0, input_shape=(4, 4, 512))(res6_2)
    res6_3 = Residual_block(mask6_3, 512, (3, 3), strides=(1, 1), dilation=3, padding="same")
    mask6_4 = Masking(mask_value=0.0, input_shape=(4, 4, 512))(res6_3)
    res6_4 = Residual_block(mask6_4, 512, (3, 3), strides=(1, 1), dilation=4, padding="same")

    mask6_5 = Masking(mask_value=0.0, input_shape=(4, 4, 512))(res6_4)
    conv6 = Conv2D(feature_size, (3, 3), strides=(1, 1), padding="same")(mask6_5)
    bn6 = LeakyReLU()(BatchNormalization()(conv6, training=True))
    encode = bn6

    mask7 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(encode)
    fu1 = WaveletUnitLayer(feature_size, name="wave1")(mask7)
    mask8 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(fu1)
    fu2 = WaveletUnitLayer(feature_size, name="wave2")(mask8)
    add1 = tf.math.add(fu2, mask7)
    addbn1 = LeakyReLU()(BatchNormalization(name="addbn1")(add1))

    mask9 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(addbn1)
    fu3 = WaveletUnitLayer(feature_size, name="wave3")(mask9)
    mask10 = Masking(mask_value=0.0, input_shape=(4, 4, feature_size))(fu3)
    fu4 = WaveletUnitLayer(feature_size, name="wave4")(mask10)
    add2 = tf.math.add(fu4, mask9)
    addbn2 = LeakyReLU()(BatchNormalization(name="addbn2")(add2))

    decode_input1 = addbn2
    deconv1 = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding="same")(decode_input1)
    debn1 = LeakyReLU()(BatchNormalization(name="debn1")(deconv1))

    decode_input2_1 = tf.math.add(debn1, mask6_5)
    deconv2_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(decode_input2_1)
    debn2_1 = LeakyReLU()(BatchNormalization(name="debn2_1")(deconv2_1))

    decode_input2_2 = tf.math.add(debn2_1, mask6_1)
    deconv2_2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(decode_input2_2)
    debn2_2 = LeakyReLU()(BatchNormalization(name="debn2_2")(deconv2_2))

    decode_input3_1 = tf.math.add(debn2_2, mask5_1)
    deconv3_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(decode_input3_1)
    debn3_1 = LeakyReLU()(BatchNormalization(name="debn3_1")(deconv3_1))

    decode_input3_2 = tf.math.add(debn3_1, mask4_1)
    deconv3_2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(decode_input3_2)
    debn3_2 = LeakyReLU()(BatchNormalization(name="debn3_2")(deconv3_2))

    decode_input4 = tf.math.add(debn3_2, mask3_1)
    deconv4 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(decode_input4)
    debn4 = LeakyReLU()(BatchNormalization(name="debn4")(deconv4))

    #decode_input5 = Add_and_Norm(debn4, mask2_1)
    decode_input5 = tf.math.add(debn4, mask2_1)
    deconv5 = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(decode_input5)
    debn5 = LeakyReLU()(BatchNormalization(name="debn5")(deconv5))

    #decode_input6 = Add_and_Norm(debn5, mask1_1)
    #decode_input6 = tf.math.add(debn5, mask1_1)
    deconv6 = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding="same")(debn5)
    debnf = LeakyReLU()(BatchNormalization(name="debn6")(deconv6))

    recon = tanh(debnf)
    return inputs, recon


def WLencoder_test4(inputs, feature_size=512):
    inputs = Input(inputs, dtype="float32")
    mask1_1 = Masking(mask_value=0, input_shape=(256, 256, 3))(inputs)
    res1_1 = Residual_Gblock(mask1_1, 32, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask1_2 = Masking(mask_value=0, input_shape=(128, 128, 16))(res1_1)
    res1_2 = Residual_Gblock(mask1_2, 32, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask2_1 = Masking(mask_value=0, input_shape=(128, 128, 16))(res1_2)
    res2_1 = Residual_Gblock(mask2_1, 64, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask2_2 = Masking(mask_value=0, input_shape=(64, 64, 32))(res2_1)
    res2_2 = Residual_Gblock(mask2_2, 64, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask3_1 = Masking(mask_value=0, input_shape=(64, 64, 32))(res2_2)
    res3_1 = Residual_Gblock(mask3_1, 128, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask3_2 = Masking(mask_value=0, input_shape=(32, 32, 64))(res3_1)
    res3_2 = Residual_Gblock(mask3_2, 128, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask4_1 = Masking(mask_value=0, input_shape=(32, 32, 64))(res3_2)
    res4_1 = Residual_Gblock(mask4_1, 256, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask4_2 = Masking(mask_value=0, input_shape=(16, 16, 128))(res4_1)
    res4_2 = Residual_Gblock(mask4_2, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask4_3 = Masking(mask_value=0, input_shape=(16, 16, 128))(res4_2)
    res4_3 = Residual_Gblock(mask4_3, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask5_1 = Masking(mask_value=0, input_shape=(16, 16, 128))(res4_3)
    res5_1 = Residual_Gblock(mask5_1, 512, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask5_2 = Masking(mask_value=0, input_shape=(8, 8, 256))(res5_1)
    res5_2 = Residual_Gblock(mask5_2, 512, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask5_3 = Masking(mask_value=0, input_shape=(8, 8, 256))(res5_2)
    res5_3 = Residual_Gblock(mask5_3, 512, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask6_1 = Masking(mask_value=0, input_shape=(8, 8, 256))(res5_3)
    res6_1 = Residual_Gblock(mask6_1, 512, (3, 3), strides=(1, 1), dilation=1, padding="same")
    mask6_2 = Masking(mask_value=0, input_shape=(4, 4, 512))(res6_1)
    res6_2 = Residual_Gblock(mask6_2, 512, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask6_3 = Masking(mask_value=0, input_shape=(4, 4, 512))(res6_2)
    res6_3 = Residual_Gblock(mask6_3, 512, (3, 3), strides=(1, 1), dilation=2, padding="same")

    mask6_4 = Masking(mask_value=0, input_shape=(4, 4, 512))(res6_3)
    conv6 = WSConv2D(feature_size, (3, 3), strides=(1, 1), padding="same")(mask6_4)
    bn6 = LeakyReLU()(GroupNormalization()(conv6))

    encode = wavelet(bn6)
    mask7 = Masking(mask_value=0.0, input_shape=(4, 4, 4*feature_size))(encode)
    fu1 = WaveletUnitLayer(4*feature_size, name="wave1")(mask7)
    mask8 = Masking(mask_value=0.0, input_shape=(4, 4, 4*feature_size))(fu1)
    fu2 = WaveletUnitLayer(4*feature_size, name="wave2")(mask8)
    add1 = tf.math.add(fu2, mask7)
    addbn1 = LeakyReLU()(GroupNormalization(name="addbn1")(add1))

    mask9 = Masking(mask_value=0.0, input_shape=(4, 4, 4*feature_size))(addbn1)
    fu3 = WaveletUnitLayer(4*feature_size, name="wave3")(mask9)
    mask10 = Masking(mask_value=0.0, input_shape=(4, 4, 4*feature_size))(fu3)
    fu4 = WaveletUnitLayer(4*feature_size, name="wave4")(mask10)
    add2 = tf.math.add(fu4, mask9)
    addbn2 = LeakyReLU()(GroupNormalization(name="addbn2")(add2))
    decode_input1 = wavelet_i(addbn2)

    debn1 = LeakyReLU()(GroupNormalization(name="debn1")(decode_input1))
    deconv1 = WSConv2DTranspose(512, (3, 3), strides=(1, 1), padding="same")(debn1)

    decode_input2_1 = tf.math.add(deconv1, mask6_3)
    debn2_1 = LeakyReLU()(GroupNormalization(name="debn2_1")(decode_input2_1))
    deconv2_1 = WSConv2DTranspose(512, (3, 3), strides=(1, 1), padding="same")(debn2_1)

    decode_input2_2 = tf.math.add(deconv2_1, mask6_1)
    debn2_2 = LeakyReLU()(GroupNormalization(name="debn2_2")(decode_input2_2))
    deconv2_2 = WSConv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(debn2_2)


    decode_input3_1 = tf.math.add(deconv2_2, mask5_1)
    debn3_1 = LeakyReLU()(GroupNormalization(name="debn3_1")(decode_input3_1))
    deconv3_1 = WSConv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(debn3_1)

    decode_input3_2 = tf.math.add(deconv3_1, mask4_1)
    debn3_2 = LeakyReLU()(GroupNormalization(name="debn3_2")(decode_input3_2))
    deconv3_2 = WSConv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(debn3_2)

    decode_input4 = tf.math.add(deconv3_2, mask3_1)
    debn4 = LeakyReLU()(GroupNormalization(name="debn4")(decode_input4))
    deconv4 = WSConv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(debn4)

    decode_input5 = tf.math.add(deconv4, mask2_1)
    debn5 = LeakyReLU()(GroupNormalization(name="debn5")(decode_input5))
    deconv5 = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(debn5)

    #decode_input6 = tf.math.add(debn5, mask1_1)
    deconv6 = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding="same")(deconv5)
    #debn6 =  LeakyReLU()(GroupNormalization(groups=1, name="debn6")(deconv6))

    recon = tanh(deconv6)
    return inputs, recon

#def hsv_silcing(hsv, size=[4, 512, 512, 1]):#第一個channel與batch_size大小一致
#    Saturation = tf.slice(hsv, [0, 0, 0, 1], size)
#    Lightness = tf.slice(hsv, [0, 0, 0, 2], size)
#    return Saturation, Lightness

def WLencoder_test5(inputs, feature_size=512, input_shape=512):
    #hsv_attend
    inputs = Input(inputs, dtype="float32")
    #hsv_inputs = tf.keras.layers.Lambda(tf.image.rgb_to_hsv)(inputs)
    hsv0 = tf.concat([tf.add(0.3*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer1")(inputs), [0, 0, 0, 1], [4, 512, 512, 1]),
                             0.7*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer2")(inputs), [0, 0, 0, 2], [4, 512, 512, 1])),
                     tf.add(0.3*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer3")(inputs), [0, 0, 0, 1], [4, 512, 512, 1]),
                            0.7*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer4")(inputs), [0, 0, 0, 2], [4, 512, 512, 1])),
                     tf.add(0.3*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer5")(inputs), [0, 0, 0, 1], [4, 512, 512, 1]),
                            0.7*tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv, name="transfer6")(inputs), [0, 0, 0, 2], [4, 512, 512, 1]))], axis=-1)
    #hsv_mask1_1 = Masking(mask_value=0, input_shape=(input_shape, input_shape, 3))(hsv0)
    hsv_res1_1 = Residual_Gblock(hsv0, 32, (3, 3), strides=(2, 2), dilation=1, padding="same")
    #hsv_mask1_2 = Masking(mask_value=0, input_shape=(int(input_shape/2), int(input_shape/2), 32))(hsv_res1_1)
    hsv_res1_2 = Residual_Gblock(hsv_res1_1, 32, (3, 3), strides=(1, 1), dilation=2, padding="same")
    #Saturation1, Lightness1 = hsv_silcing(hsv_res1_2)
    #hsv1 = tf.concat([tf.add(0.8*Saturation1, 0.2*Lightness1),
    #                 tf.add(0.8*Saturation1, 0.2*Lightness1),
    #                 tf.add(0.8*Saturation1, 0.2*Lightness1)], axis=-1)

    #hsv_mask2_1 = Masking(mask_value=0, input_shape=(int(input_shape/2), int(input_shape/2), 32))(hsv_res1_2)
    hsv_res2_1 = Residual_Gblock(hsv_res1_2, 64, (3, 3), strides=(2, 2), dilation=1, padding="same")
    #hsv_mask2_2 = Masking(mask_value=0, input_shape=(int(input_shape/4), int(input_shape/4), 64))(hsv_res2_1)
    hsv_res2_2 = Residual_Gblock(hsv_res2_1, 64, (3, 3), strides=(1, 1), dilation=2, padding="same")
    #Saturation2, Lightness2 = hsv_silcing(hsv_res2_2)
    #hsv2 = tf.concat([tf.add(0.8*Saturation2, 0.2*Lightness2),
    #                 tf.add(0.8*Saturation2, 0.2*Lightness2),
    #                 tf.add(0.8*Saturation2, 0.2*Lightness2)], axis=-1)

    #hsv_mask3_1 = Masking(mask_value=0, input_shape=(int(input_shape/4), int(input_shape/4), 64))(hsv_res2_2)
    hsv_res3_1 = Residual_Gblock(hsv_res2_2, 128, (3, 3), strides=(2, 2), dilation=1, padding="same")
    #hsv_mask3_2 = Masking(mask_value=0, input_shape=(int(input_shape/8), int(input_shape/8), 128))(hsv_res3_1)
    hsv_res3_2 = Residual_Gblock(hsv_res3_1, 128, (3, 3), strides=(1, 1), dilation=2, padding="same")
    #Saturation3, Lightness3 = hsv_silcing(hsv_res3_2)
    #hsv3 = tf.concat([tf.add(0.8*Saturation3, 0.2*Lightness3),
    #                 tf.add(0.8*Saturation3, 0.2*Lightness3),
    #                 tf.add(0.8*Saturation3, 0.2*Lightness3)], axis=-1)

    #hsv_mask4_1 = Masking(mask_value=0, input_shape=(int(input_shape/8), int(input_shape/8), 128))(hsv_res3_2)
    hsv_res4_1 = Residual_Gblock(hsv_res3_2, 256, (3, 3), strides=(2, 2), dilation=1, padding="same")
    #hsv_mask4_2 = Masking(mask_value=0, input_shape=(int(input_shape/16), int(input_shape/16), 256))(hsv_res4_1)
    hsv_res4_2 = Residual_Gblock(hsv_res4_1, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")
    #hsv_mask4_3 = Masking(mask_value=0, input_shape=(int(input_shape/16), int(input_shape/16), 256))(hsv_res4_2)
    hsv_res4_3 = Residual_Gblock(hsv_res4_2, 256, (3, 3), strides=(1, 1), dilation=1, padding="same")
    #Saturation4, Lightness4 = hsv_silcing(hsv_res4_3)
    #hsv4 = tf.concat([tf.add(0.8*Saturation4, 0.2*Lightness4),
    #                 tf.add(0.8*Saturation4, 0.2*Lightness4),
    #                 tf.add(0.8*Saturation4, 0.2*Lightness4)], axis=-1)

    #main
    #inputs0 = tf.keras.layers.Attention()([inputs, hsv0])
    inputs0 = tf.math.multiply(inputs, hsv0)
    mask1_1 = Masking(mask_value=0, input_shape=(input_shape, input_shape, 3))(inputs0)
    res1_1 = Residual_Gblock(mask1_1, 32, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask1_2 = Masking(mask_value=0, input_shape=(int(input_shape/2), int(input_shape/2), 32))(res1_1)
    res1_2 = Residual_Gblock(mask1_2, 32, (3, 3), strides=(1, 1), dilation=2, padding="same")

    #inputs1 = tf.keras.layers.Attention()([res1_2, hsv_res1_2])
    inputs1 = tf.math.multiply(res1_2, hsv_res1_2)
    mask2_1 = Masking(mask_value=0, input_shape=(int(input_shape/2), int(input_shape/2), 32))(inputs1)
    res2_1 = Residual_Gblock(mask2_1, 64, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask2_2 = Masking(mask_value=0, input_shape=(int(input_shape/4), int(input_shape/4), 64))(res2_1)
    res2_2 = Residual_Gblock(mask2_2, 64, (3, 3), strides=(1, 1), dilation=2, padding="same")

    #inputs2 = tf.keras.layers.Attention()([res2_2, hsv_res2_2])
    inputs2 = tf.math.multiply(res2_2, hsv_res2_2)
    mask3_1 = Masking(mask_value=0, input_shape=(int(input_shape/4), int(input_shape/4), 64))(inputs2)
    res3_1 = Residual_Gblock(mask3_1, 128, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask3_2 = Masking(mask_value=0, input_shape=(int(input_shape/8), int(input_shape/8), 128))(res3_1)
    res3_2 = Residual_Gblock(mask3_2, 128, (3, 3), strides=(1, 1), dilation=2, padding="same")

    #inputs3 = tf.keras.layers.Attention()([res3_2, hsv_res3_2])
    inputs3 = tf.math.multiply(res3_2, hsv_res3_2)
    mask4_1 = Masking(mask_value=0, input_shape=(int(input_shape/8), int(input_shape/8), 128))(inputs3)
    res4_1 = Residual_Gblock(mask4_1, 256, (3, 3), strides=(2, 2), dilation=1, padding="same")
    mask4_2 = Masking(mask_value=0, input_shape=(int(input_shape/16), int(input_shape/16), 256))(res4_1)
    res4_2 = Residual_Gblock(mask4_2, 256, (3, 3), strides=(1, 1), dilation=2, padding="same")
    mask4_3 = Masking(mask_value=0, input_shape=(int(input_shape/16), int(input_shape/16), 256))(res4_2)
    res4_3 = Residual_Gblock(mask4_3, 256, (3, 3), strides=(1, 1), dilation=1, padding="same")

    #inputs4 = tf.keras.layers.Attention()([res4_3, hsv_res4_3])
    inputs4 = tf.math.multiply(res4_3, hsv_res4_3)
    mask5_1 = Masking(mask_value=0, input_shape=(int(input_shape/16), int(input_shape/16), 256))(inputs4)
    conv5 = WSConv2D(feature_size, (3, 3), strides=(2, 2), padding="same")(mask5_1)
    bn6 = LeakyReLU()(GroupNormalization()(conv5))

    encode = wavelet(bn6)
    maskw1 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(encode)
    fu1 = WaveletUnitLayer(4*feature_size, name="wave1")(maskw1)
    maskw2 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(fu1)
    fu2 = WaveletUnitLayer(4*feature_size, name="wave2")(maskw2)
    add1 = tf.math.add(fu2, maskw1)
    addbn1 = LeakyReLU()(GroupNormalization(name="addbn1")(add1))

    maskw3 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(addbn1)
    fu3 = WaveletUnitLayer(4*feature_size, name="wave3")(maskw3)
    maskw4 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(fu3)
    fu4 = WaveletUnitLayer(4*feature_size, name="wave4")(maskw4)
    add2 = tf.math.add(fu4, maskw3)
    addbn2 = LeakyReLU()(GroupNormalization(name="addbn2")(add2))

    #maskw5 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(addbn2)
    #fu5 = WaveletUnitLayer(4*feature_size, name="wave5")(maskw5)
    #maskw6 = Masking(mask_value=0.0, input_shape=(int(input_shape/32), int(input_shape/32), 4*feature_size))(fu5)
    #fu6 = WaveletUnitLayer(4*feature_size, name="wave6")(maskw6)
    #add3 = tf.math.add(fu6, maskw5)
    #addbn3 = LeakyReLU()(GroupNormalization(name="addbn3")(add3))
    decode = wavelet_i(addbn2)
    #print(decode)


    decode_input1 = decode
    debn1 = LeakyReLU()(GroupNormalization(name="debn1")(decode_input1))
    deconv1 = WSConv2DTranspose(512, (3, 3), strides=(1, 1), padding="same")(debn1)

    decode_input2 = tf.math.add(deconv1, bn6)
    debn2 = LeakyReLU()(GroupNormalization(name="debn2_2")(decode_input2))
    deconv2 = WSConv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(debn2)


    decode_input3 = tf.math.add(deconv2, res4_3)
    debn3 = LeakyReLU()(GroupNormalization(name="debn3_1")(decode_input3))
    deconv3 = WSConv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(debn3)

    decode_input4 = tf.math.add(deconv3, res3_2)
    debn4 = LeakyReLU()(GroupNormalization(name="debn3_2")(decode_input4))
    deconv4 = WSConv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(debn4)

    decode_input5 = tf.math.add(deconv4, res2_2)
    debn5 = LeakyReLU()(GroupNormalization(name="debn4")(decode_input5))
    deconv5 = WSConv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(debn5)

    #decode_input6 = tf.math.add(debn5, res1_2)
    deconv6 = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(deconv5)
    #debn6 =  LeakyReLU()(GroupNormalization(groups=1, name="debn6")(deconv6))

    recon = tanh(deconv6)
    return inputs, recon


def discriminator(inputs):
    inputs = Input(inputs, dtype="float32")
    conv1_1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name="conv1")(inputs)
    relu1_1 = LeakyReLU()(BatchNormalization(name="gn1")(conv1_1))
    conv2_1 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name="conv2")(relu1_1)
    relu2_1 = LeakyReLU()(BatchNormalization(name="gn2")(conv2_1))
    conv3_1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name="conv3")(relu2_1)
    relu3_1 = LeakyReLU()(BatchNormalization(name="gn3")(conv3_1))
    conv4_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name="conv4")(relu3_1)
    relu4_1 = LeakyReLU()(BatchNormalization(name="gn4")(conv4_1))
    conv5_1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name="conv5")(relu4_1)
    relu5_1 = LeakyReLU()(BatchNormalization(name="gn5")(conv5_1))

    flatten = Flatten()(relu5_1)
    fc = Dense(1)(flatten)
    return inputs, fc


def build(input_shape):
    inputs, outputs = FQencoder(input_shape)
    d_inputs, d_outputs = discriminator(input_shape)
    inputs_w, outputs_w = WLencoder_test5(input_shape)
    model = tf.keras.Model(inputs, outputs, name="FQencoder")
    d = tf.keras.Model(d_inputs, d_outputs, name="Discriminator")
    model_w = tf.keras.Model(inputs_w, outputs_w, name="WLencoder")
    return model, d, model_w

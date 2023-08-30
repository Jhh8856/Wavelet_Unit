import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, BatchNormalization

batch_size = 4 #一致

def batch_norm(x):
    m_h, v_h = tf.nn.moments(x, [0])
    h_batch_norm = tf.nn.batch_normalization(x, mean=m_h, variance=v_h,
                                             offset=tf.constant(0.),
                                             scale=tf.constant(1.),
                                             variance_epsilon=1e-8)
    return h_batch_norm
#https://github.com/taki0112/Group_Normalization-Tensorflow
def group_norm(x, gamma, beta, G=32, epsilon=1e-5):
    N, C, H, W = x.shape
    N = batch_size
    G = min(G, C)

    x = tf.reshape(x, [N, H, W, G, C // G])
    mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + epsilon)

    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN

def Weight_Standardization(kernel, epsilon=1e-5):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name="kernel_mean")
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + epsilon)
    return kernel

def Add_and_Norm(feature1, feature2):
    temp = tf.math.add(feature1, feature2)
    temp = BatchNormalization()(temp)
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
def WaveletUnit(feature, r1, r2, r1_2, r2_2, r3, r3_2, r4, r4_2, r5, r5_2, r6, r7):
    from wavetf import WaveTFFactory
    w = WaveTFFactory().build('db2', dim=2)
    w_i = WaveTFFactory().build('db2', dim=2, inverse=True)
    # print(feature)
    r1 = Weight_Standardization(r1)
    r2 = Weight_Standardization(r2)
    r1_2 = Weight_Standardization(r1_2)
    r2_2 = Weight_Standardization(r2_2)
    r3 = Weight_Standardization(r3)
    r3_2 = Weight_Standardization(r3_2)
    r4 = Weight_Standardization(r4)
    r4_2 = Weight_Standardization(r4_2)
    r5 = Weight_Standardization(r5)
    r5_2 = Weight_Standardization(r5_2)
    r6 = Weight_Standardization(r6)
    r7 = Weight_Standardization(r7)

    _, y_r1, y_r2, y_i = tf.split(feature, num_or_size_splits=4, axis=-1)  # fixing
    y_r = tf.math.add(y_r1, y_r2)
    y_r = batch_norm(y_r)
    #r = tf.random.normal([1, 1, tf.shape(y_r)[-1], tf.shape(y_r)[-1]])

    y_r_nor = tf.nn.conv2d(y_r, r1, strides=1,
                           padding="SAME", name="y_r_nor")
    y_r_nor = group_norm(y_r_nor,
                         tf.ones([1, 1, 1, tf.shape(y_r_nor)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_r_nor)[-1]]))
    y_r_nor = tf.nn.leaky_relu(y_r_nor)

    #r2 = tf.random.normal([1, 1, tf.shape(y_i)[-1], tf.shape(y_i)[-1]])
    y_i_nor = tf.nn.conv2d(y_i, r2, strides=1,
                           padding="SAME", name="y_i_nor")
    y_i_nor = group_norm(y_i_nor,
                         tf.ones([1, 1, 1, tf.shape(y_i_nor)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_i_nor)[-1]]))
    y_i_nor = tf.nn.leaky_relu(y_i_nor)

    #r1_3x3 = tf.random.normal([1, 1, tf.shape(y_r)[-1], tf.shape(y_r)[-1]])
    y_r3x3 = tf.nn.conv2d(y_r_nor, r1_2, strides=1,
                          padding="SAME", name="y_r3x3")
    y_r3x3 = group_norm(y_r3x3,
                        tf.ones([1, 1, 1, tf.shape(y_r3x3)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_r3x3)[-1]]))
    y_r3x3 = tf.nn.leaky_relu(y_r3x3)

    #r2_3x3 = tf.random.normal([1, 1, tf.shape(y_i)[-1], tf.shape(y_i)[-1]])
    y_i3x3 = tf.nn.conv2d(y_i_nor, r2_2, strides=1,
                          padding="SAME", name="y_i3x3")
    y_i3x3 = group_norm(y_i3x3,
                        tf.ones([1, 1, 1, tf.shape(y_i3x3)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_i3x3)[-1]]))
    y_i3x3 = tf.nn.leaky_relu(y_i3x3)


    freq_y_i = w(y_i_nor)
    #print(freq_y_i)
    freq_y_i_low, freq_y_i_mid1, freq_y_i_mid2, freq_y_i_high = tf.split(freq_y_i, num_or_size_splits=4, axis=-1)
    #freq_y_i_mid = tf.concat((freq_y_i_mid1, freq_y_i_mid2), axis=-1)
    print(freq_y_i_low)
    print(freq_y_i_mid1)
    print(freq_y_i_high)

    y_sm1 = freq_y_i_mid1
    #r3 = tf.random.normal([1, 1, tf.shape(y_sm)[-1], tf.shape(y_sm)[-1]])
    y_sm1 = tf.nn.conv2d(y_sm1, r3, strides=1, padding="SAME", name="specm1")
    y_sm1 = tf.nn.conv2d(y_sm1, r3_2, strides=1, padding="SAME", name="specm1_2")
    y_sm1 = group_norm(y_sm1,
                      tf.ones([1, 1, 1, tf.shape(y_sm1)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_sm1)[-1]]))
    y_sm1 = tf.nn.leaky_relu(y_sm1)

    y_sm2 = freq_y_i_mid1
    # r3 = tf.random.normal([1, 1, tf.shape(y_sm)[-1], tf.shape(y_sm)[-1]])
    y_sm2 = tf.nn.conv2d(y_sm2, r4, strides=1, padding="SAME", name="specm2")
    y_sm2 = tf.nn.conv2d(y_sm2, r4_2, strides=1, padding="SAME", name="specm2_2")
    y_sm2 = group_norm(y_sm2,
                       tf.ones([1, 1, 1, tf.shape(y_sm2)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_sm2)[-1]]))
    y_sm2 = tf.nn.leaky_relu(y_sm2)

    y_sh = freq_y_i_high
    #r4 = tf.random.normal([1, 1, tf.shape(y_sh)[-1], tf.shape(y_sh)[-1]])
    y_sh = tf.nn.conv2d(y_sh, r5, strides=1, padding="SAME", name="spech")
    y_sh = tf.nn.conv2d(y_sh, r5_2, strides=1, padding="SAME", name="spech_2")
    y_sh = group_norm(y_sh,
                      tf.ones([1, 1, 1, tf.shape(y_sh)[-1]]), tf.zeros([1, 1, 1, tf.shape(y_sh)[-1]]))
    y_sh = tf.nn.leaky_relu(y_sh)

    print(freq_y_i_low)
    print(y_sm1)
    print(y_sh)
    y_s = tf.concat((freq_y_i_low, y_sm1, y_sm2, y_sh), axis=-1)
    print(y_s)
    # z_r, z_i = tf.split(y_s, num_or_size_splits=2, axis=-1)
    # z = tf.dtypes.complex(z_r, z_i)
    # z = tf.signal.irfft2d(z)
    #w_i = WaveTFFactory().build('db2', dim=2, inverse=True)
    z = w_i(y_s)
    #r5 = tf.random.normal([1, 1, tf.shape(z)[-1], tf.shape(z)[-1]])
    spectral = tf.nn.conv2d(z, r6, strides=1, padding="SAME", name="spectral")
    spectral = group_norm(spectral,
                          tf.ones([1, 1, 1, tf.shape(spectral)[-1]]), tf.zeros([1, 1, 1, tf.shape(spectral)[-1]]))
    spectral = tf.nn.leaky_relu(spectral)

    global_feature = tf.add(y_r3x3, spectral)
    global_feature = group_norm(spectral,
                                tf.ones([1, 1, 1, tf.shape(global_feature)[-1]]), tf.zeros([1, 1, 1, tf.shape(global_feature)[-1]]))
    global_feature = tf.nn.leaky_relu(global_feature)

    local_feature = tf.add(y_r3x3, y_i3x3)
    local_feature = group_norm(local_feature,
                               tf.ones([1, 1, 1, tf.shape(local_feature)[-1]]), tf.zeros([1, 1, 1, tf.shape(local_feature)[-1]]))
    local_feature = tf.nn.leaky_relu(local_feature)

    final_a = tf.concat((global_feature, local_feature), axis=-1)
    final_conv = tf.nn.conv2d(final_a, r7, strides=1, padding="SAME", name="final")
    final_bn = group_norm(final_conv,
                          tf.ones([1, 1, 1, tf.shape(final_conv)[-1]]), tf.zeros([1, 1, 1, tf.shape(final_conv)[-1]]))
    final = tf.nn.leaky_relu(final_bn)

    return final


#class WaveletUnitLayer(keras.layers.Layer):
#    def __init__(self, units=32, input_dim=32):
#        super(WaveletUnitLayer, self).__init__()
#        w_init = tf.random_normal_initializer()
#        self.units = units
#        self.input_dim = input_dim
#        self.w = tf.Variable(
#            initial_value=w_init(shape=(self.input_dim, self.units), dtype="float32"),
#            trainable=True,
#        )
#        b_init = tf.zeros_initializer()
#        self.b = tf.Variable(
#            initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=True
#        )
#
#    def call(self, inputs):
#        return WaveletUnit(tf.matmul(inputs, self.w)) + self.b
#
#    def get_config(self):
#        config = super().get_config()
#        config.update({
#            "units": self.units,
#            "input_dim": self.input_dim,
#            "w": self.w,
#            "b": self.b
#        })
#        return config

#    @classmethod
#    def from_config(cls, config):
#        return cls(**config)

class WaveletUnitLayer(keras.layers.Layer):
    def __init__(self, units=512, **kwargs):
        super(WaveletUnitLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        #self.w = self.add_weight(name='w',
        #    shape=(input_shape[-1], self.units),
        #    initializer="random_normal",
        #    trainable=True,
        #)
        self.b = self.add_weight(name='b',
            shape=(self.units,), initializer="random_normal", trainable=True
        )

        self.r1 = self.add_weight(name='r1',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r2 = self.add_weight(name='r2',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r1_2 = self.add_weight(name='r1_2',
            shape=(3, 3, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r2_2 = self.add_weight(name='r2_2',
            shape=(3, 3, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r3 = self.add_weight(name='r3',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r3_2 = self.add_weight(name='r3_2',
                                  shape=(1, 1, int((input_shape[-1]) / 4), int((input_shape[-1]) / 4)),
                                  initializer="random_normal",
                                  trainable=True,
                                  )
        self.r4 = self.add_weight(name='r4',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r4_2 = self.add_weight(name='r4_2',
                                  shape=(1, 1, int((input_shape[-1]) / 4), int((input_shape[-1]) / 4)),
                                  initializer="random_normal",
                                  trainable=True,
                                  )
        self.r5 = self.add_weight(name='r5',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r5_2 = self.add_weight(name='r5_2',
                                  shape=(1, 1, int((input_shape[-1]) / 4), int((input_shape[-1]) / 4)),
                                  initializer="random_normal",
                                  trainable=True,
                                  )
        self.r6 = self.add_weight(name='r6',
            shape=(1, 1, int((input_shape[-1])/4), int((input_shape[-1])/4)),
            initializer="random_normal",
            trainable=True,
        )
        self.r7 = self.add_weight(name='r7',
            shape=(1, 1, int((input_shape[-1])/2), int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )


    def call(self, inputs):
        return tf.add(WaveletUnit(inputs, self.r1, self.r2, self.r1_2, self.r2_2, self.r3, self.r3_2,
                                  self.r4, self.r4_2, self.r5, self.r5_2, self.r6, self.r7),
                      self.b)

    def get_config(self):
        config = super(WaveletUnitLayer, self).get_config()
        config.update({"units": self.units})
        return config


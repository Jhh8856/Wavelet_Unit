import numpy as np
import scipy.stats as st
import tensorflow as tf
from WaveletUnit import WaveletUnit


def L1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(tf.math.subtract(y_true, y_pred)))

def L2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(tf.math.subtract(y_true, y_pred)))

def wavelet_loss(y_true, y_pred):
    from wavetf import WaveTFFactory
    w = WaveTFFactory().build('db2', dim=2)
    a = w(y_true)
    b = w(y_pred)
    return L1_loss(a, b)

def wavelet_conv_loss(y_true, y_pred):
    from wavetf import WaveTFFactory
    w = WaveTFFactory().build('db2', dim=2)
    token = tf.random.normal([7, 7, 3, 64])

    y_t = w(y_true)
    y_tc = tf.nn.conv2d(y_t, token, strides=1, padding="SAME")
    y_p = w(y_pred)
    y_pc = tf.nn.conv2d(y_p, token, strides=1, padding="SAME")

    return L1_loss(y_pc, y_tc)

def gaussian_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter

def make_gauss_var(name, size, sigma, c_i):
    with tf.device("/cpu:0"):
        kernel = gaussian_kernel(size, sigma, c_i)
        var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
    return var

def smooth(input, padding='SAME'):
        # Get the number of channels in the input
        #c_i = input.get_shape().as_list()[3]
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                             padding=padding)

        #kernel = make_gauss_var('gauss_weight', 256, 4, 3)
        kernel = gaussian_kernel(256, 4, 3)
        output = convolve(input, kernel)
        return output

def color_loss(y_true, y_pred):
    true = smooth(y_true)
    pred = smooth(y_pred)
    return L2_loss(true, pred)

#https://github.com/mtyka/laploss/blob/master/laploss.py
#def gauss_kernel(size=5, sigma=1.0):
#  grid = np.float32(np.mgrid[0:size,0:size].T)
#  gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
#  kernel = np.sum(gaussian(grid), axis=2)
#  kernel /= np.sum(kernel)
#  return kernel

#def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
#  t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),
#                        [k_size, k_size, 1, 1])
#  t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
#  t_result = t_input
#  for r in range(repeats):
#    t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
#        strides=[1, stride, stride, 1], padding='SAME')
#  return t_result

#def make_laplacian_pyramid(t_img, max_levels=3):
#  t_pyr = []
#  current = t_img
#  for level in range(max_levels):
#    t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
#    t_diff = current - t_gauss
#    t_pyr.append(t_diff)
#    current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
#  t_pyr.append(current)
#  return t_pyr

#def laploss(y_true, y_pred):
#  t_pyr1 = make_laplacian_pyramid(y_true)
#  t_pyr2 = make_laplacian_pyramid(y_pred)
#  t_losses = [tf.norm(a-b,ord=1)/tf.size(a, out_type=tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
#  t_loss = tf.reduce_sum(t_losses)*tf.shape(y_true, out_type=tf.float32)[0]

#  return t_loss

#unuse
def Perceptual_loss_VGG16(y_true, y_pred):
    data_dict_vgg16 = np.load("./weight/vgg16.npy", encoding='latin1', allow_pickle=True).item()
    y_true = tf.image.resize(y_true, [224, 224])
    y_pred = tf.image.resize(y_pred, [224, 224])
    def get_conv_filter(name):
        return tf.constant(data_dict_vgg16[name][0], name="filter")

    def get_bias(name):
        return tf.constant(data_dict_vgg16[name][1], name="biases")

    def get_fc_weight(name):
        return tf.constant(data_dict_vgg16[name][0], name="weights")

    def conv_layer(bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(bottom, name):
        with tf.compat.v1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = get_fc_weight(name)
            biases = get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def vgg16(img):

        conv1_1 = conv_layer(img, "conv1_1")
        conv1_2 = conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1")
        conv2_2 = conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1")
        conv3_2 = conv_layer(conv3_1, "conv3_2")
        conv3_3 = conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1")
        conv4_2 = conv_layer(conv4_1, "conv4_2")
        conv4_3 = conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1")
        conv5_2 = conv_layer(conv5_1, "conv5_2")
        conv5_3 = conv_layer(conv5_2, "conv5_3")
        pool5 = max_pool(conv5_3, 'pool5')

        fc6 = fc_layer(pool5, "fc6")
        #assert self.fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        fc8 = fc_layer(relu7, "fc8")

        prob = tf.nn.softmax(fc8, name="prob")

        #return prob
        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    f_p1, f_p2, f_p3, f_p4, f_p5 = vgg16(y_pred)
    f_t1, f_t2, f_t3, f_t4, f_t5 = vgg16(y_true)

    f1 = tf.math.square(tf.math.subtract(f_p1, f_t1))
    f2 = tf.math.square(tf.math.subtract(f_p2, f_t2))
    f3 = tf.math.square(tf.math.subtract(f_p3, f_t3))
    f4 = tf.math.square(tf.math.subtract(f_p4, f_t4))
    f5 = tf.math.square(tf.math.subtract(f_p5, f_t5))

    f = tf.reduce_sum([tf.reduce_mean(f1),
                       tf.reduce_mean(f2),
                       tf.reduce_mean(f3),
                       tf.reduce_mean(f4),
                       tf.reduce_mean(f5)])

    return f

def Feature_matching_loss(y_true, y_pred):
    d = tf.keras.models.load_model("./saved_model/temp_d.h5")
    true = d.predict(y_true)
    pred = d.predict(y_pred)
    return L2_loss(true, pred)

def dct_loss(y_ture, y_pred):
    def dct_2d(feature_map, norm='ortho'):
        X1 = tf.signal.dct(feature_map, type=2, norm=norm)
        X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
        X2 = tf.signal.dct(X1_t, type=2, norm=norm)
        X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
        return X2_t

    return L1_loss(dct_2d(y_ture), dct_2d(y_pred))

def Total_Variation_loss(y_true, y_pred):
    TV_true = tf.image.total_variation(y_true)
    TV_pred = tf.image.total_variation(y_pred)
    return tf.math.reduce_mean(tf.math.abs(tf.math.subtract(TV_true, TV_pred)))


def SSIM_loss(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1)


def Total_loss(y_true, y_pred):
    return L1_loss(y_true, y_pred) +\
           wavelet_loss(y_true, y_pred) +\
           0.1*wavelet_conv_loss(y_true, y_pred) +\
           dct_loss(y_true, y_pred) +\
           0.01*Perceptual_loss_VGG16(y_true, y_pred) +\
           0.1*color_loss(y_true, y_pred)


# metrics
def PSNR_1(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)


def PSNR_255(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)


def SSIM_1(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)


def SSIM_255(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255)
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from preprocess import Place_get_data, Celeba_HQ_get_data, Place_get_slice
from WaveletUnit import WaveletUnitLayer
from model import GroupNormalization, WSConv2D, WSConv2DTranspose
from loss import PSNR_1, SSIM_1, Total_loss, L1_loss, L2_loss, wavelet_loss, wavelet_conv_loss, color_loss
from wavetf._daubachies_conv import DaubWaveLayer2D, InvDaubWaveLayer2D

batch_size = 4
mask_type = "free"
masked = 96
image_size = 512
border = int((image_size-masked)/2)
#test_photo_num = 57
#test_photo_num = 3
#test_photo_num = 34

#test_photo_num = 86
#test_photo_num = 29
test_photo_num = 145

#test_photo_num = 25
#test_photo_num = 55

#test_photo_num = 97
#test_photo_num = 96
#test_photo_num = 95

#images_test, images_test_mask, grd_test, grd_test_mask = Place_get_data("./dataset/Place/places365_test.txt", "./dataset/Place/test_large",
#                                                                         image_shape=image_size, num=160, mask_type=mask_type, masked=masked, shuffle=True)
images_test, images_test_mask, grd_test, grd_test_mask = Place_get_slice("./dataset/Place/val",
                                                                         image_shape=image_size, num=160, mask_type=mask_type, masked=masked, samp=500)
#images_test, images_test_mask, grd_test, grd_test_mask = Celeba_HQ_get_data("./dataset/Celeba_HQ/train",
#                                                                         image_shape=image_size, num=160, mask_type=mask_type, masked=masked, samp=1000)

inpaint = tf.keras.models.load_model('saved_model/sample31',
                                    custom_objects={"WSConv2D": WSConv2D,
                                                    "WSConv2DTranspose": WSConv2DTranspose,
                                                    "WaveletUnitLayer": WaveletUnitLayer,
                                                    "DaubWaveLayer2D": DaubWaveLayer2D,
                                                    "InvDaubWaveLayer2D": InvDaubWaveLayer2D,
                                                    "GroupNormalization": GroupNormalization,
                                                    "L1_loss": L1_loss,
                                                    "L2_loss": L2_loss,
                                                    "Total_loss": Total_loss,
                                                    "PSNR_1": PSNR_1,
                                                    "SSIM_1": SSIM_1,
                                                    "wavelet_loss": wavelet_loss,
                                                    "wavelet_conv_loss": wavelet_conv_loss,
                                                    "color_loss": color_loss})

#inpaint = Model(inputs=loading.get_layer(name="WLencoder").input,
#                outputs=loading.get_layer(name="WLencoder").output)
#inpaint.load_weights("./checkpoints/sample/sample3")
inpaint.summary()
#inpaint.compile(optimizer=RMSprop(learning_rate=1e-3), loss=Total_loss)

images_test = images_test / 255.0
#images_test_mask = images_test_mask / 255.0
grd_test = grd_test / 255.0
#grd_test_mask = grd_test_mask / 255.0


result = inpaint.evaluate(images_test, grd_test, batch_size=batch_size)
#print(result)

paint = inpaint.predict(images_test, batch_size=batch_size)
#print(tf.shape(paint))

painting = paint[test_photo_num]
photo = images_test[test_photo_num]
mask = images_test_mask[test_photo_num]
photo_grd = grd_test[test_photo_num]

img0 = np.uint8(photo*255)
img1 = np.uint8(photo_grd*255)
imgp = np.uint8(painting*255)
imgh = np.uint8(tf.image.rgb_to_hsv(photo_grd)*255)
m = np.uint8(mask)
img = imgp


m[m==0] = 255
m[m!=255] = 0

img0 = Image.fromarray(img0.reshape((image_size, image_size, 3)))
img1 = Image.fromarray(img1.reshape((image_size, image_size, 3)))
imgh = Image.fromarray(imgh.reshape((image_size, image_size, 3)))
#imgp = Image.fromarray(imgp)
img = Image.fromarray(img.reshape((image_size, image_size, 3)))
m = Image.fromarray(m.reshape(image_size ,image_size, 3))


img0.save("./img0.png")
#img0.show()
img1.save("./img1.png")
#img1.show()
imgh.save("./imgh.png")
img.save("./img.png")
#img.show()
m.save("./imgm.png")
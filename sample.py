import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from preprocess import Place_get_data, Celeba_HQ_get_data, Place_get_slice
from WaveletUnit import WaveletUnitLayer
from model import FourierUnit, build, WLencoder_test, GroupNormalization, WSConv2D, WSConv2DTranspose
from loss import PSNR_1, SSIM_1, Total_loss, L1_loss, L2_loss, wavelet_loss, wavelet_conv_loss, color_loss, Feature_matching_loss, dct_loss
from wavetf._daubachies_conv import DaubWaveLayer2D, InvDaubWaveLayer2D

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*23)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


#trans_type = "Fourier"
trans_type = "Wavelet"
train_count = 1000
val_count = 100
#train_count = 10080 #必須整除batch_size
#val_count = 1008 #必須整除batch_size
samp_seed = 20000
image_size = 512
mask_type = "free"

#mask_type = "square"
masked = 96
border = int((image_size-masked)/2)

input_shape = (image_size, image_size, 3)
lr = 1e-3
pre_epochs = 100
epochs = 0
#batch_size = 16
batch_size = 4 #必須與_base_wavelets.py中一致

#target = input("model name:")
try:
    sample_w = tf.keras.models.load_model('saved_model/sample31',
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
    print("load existed model")
except:
    sample_g, sample_d, sample_w = build(input_shape=input_shape)
    print("error, build a new model")
#print("load model: {}".format(target))


#images, images_mask, grd, grd_mask = Place_get_data("./dataset/Place/places365_train_standard.txt", "./dataset/Place/data_large",
#                                                     image_shape=image_size, num=train_count, mask_type=mask_type, masked=masked, shuffle=True, samp=samp_seed)
#images_val, images_val_mask, grd_val, grd_val_mask = Place2_get_data("./dataset/Place/places365_val.txt", "./dataset/Place/val_large",
#                                                                     image_shape=image_size, num=val_count, mask_type=mask_type, masked=masked, shuffle=True, samp=samp_seed)

images, images_mask, grd, grd_mask = Place_get_slice("./dataset/Place/train",
                                                     image_shape=image_size, num=train_count, mask_type=mask_type, masked=masked, shuffle=True, samp=samp_seed)
images_val, images_val_mask, grd_val, grd_val_mask = Place_get_slice("./dataset/Place/val",
                                                                     image_shape=image_size, num=val_count, mask_type=mask_type, masked=masked, shuffle=True, samp=500)

#images, images_mask, grd, grd_mask = Celeba_HQ_get_data("./dataset/Celeba_HQ/train",
#                                                     image_shape=image_size, num=train_count, mask_type=mask_type, masked=masked, shuffle=True, samp=samp_seed)
#images_val, images_val_mask, grd_val, grd_val_mask = Celeba_HQ_get_data("./dataset/Celeba_HQ/val",
#                                                                     image_shape=image_size, num=val_count, mask_type=mask_type, masked=masked, shuffle=True, samp=100)


images = images / 255.0
#images_mask = images_mask / 255.0
grd = grd / 255.0
#grd_mask = grd_mask / 255.0
images_val = images_val / 255.0
#images_val_mask = images_val_mask / 255.0
grd_val = grd_val / 255.0
#grd_val_mask = grd_val_mask / 255.0

#sample_g, sample_d, sample_w = build(input_shape=input_shape)

psnr = PSNR_1
ssim = SSIM_1
wave = wavelet_loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#waveconv = wavelet_conv_loss
color = color_loss
if pre_epochs != 0:
    if trans_type == "Fourier":
        sample_g.compile(optimizer=RMSprop(learning_rate=lr),
                         loss=Total_loss,
                         metrics=[psnr, ssim])
        sample_g.summary()
        sample_g.fit(images, grd, batch_size=batch_size, epochs=pre_epochs,
                     validation_data=(images_val, grd_val))
        sample_g.save("./saved_model/sample.h5")
        sample_g.save_weights("./checkpoint/sample/sample")
    elif trans_type == "Wavelet":
        sample_w.compile(optimizer=RMSprop(learning_rate=lr),
                         loss=Total_loss,
                         metrics=[psnr, ssim])
        checkpoint_filepath = "./checkpoint/sample"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='PSNR_1',
            mode='max',
            save_best_only=True)

        sample_w.summary()
        sample_w.fit(images, grd, batch_size=batch_size, epochs=pre_epochs,
                     validation_data=(images_val, grd_val),
                     callbacks=[model_checkpoint_callback])
        #sample_w.save("./saved_model/sample15.h5") #256
        #sample_w.save("./saved_model/sample16.h5") #256
        #sample_w.save("./saved_model/sample17.h5") #256
        #sample_w.save("./saved_model/sample18.h5") #256
        #sample_w.save("./saved_model/sample19.h5") #celeb256
        #sample_w.save("./saved_model/sample20.h5") #celeb
        #sample_w.save("./saved_model/sample21.h5")  #celeb, G
        #sample_w.save("./saved_model/sample22.h5")  #G
        #sample_w.save("./saved_model/sample23.h5") #G, 重跑
        #sample_w.save("./saved_model/sample24.h5")  # celeb, G
        #sample_w.save("./saved_model/sample29")
        #sample_w.save("./saved_model/sample30") #celeb, HSV <-use this
        sample_w.save("./saved_model/sample31") #place, HSV
        #sample_w.save("./saved_model/sample32")
        #sample_w.save("./saved_model/sample33")


        sample_w.save_weights("./checkpoint/sample/sample31")


#sample_d.compile(optimizer=RMSprop(learning_rate=lr), loss=L1_loss)

#sample = gan_build()
#sample.summary()
#sample.compile(optimizer=RMSprop(learning_rate=lr), loss=L1_loss)

def train(X_train, y_train, epochs=50000, batch=batch_size, save_interval=100):

    def gan_build(generator, discriminator):
        discriminator.trainable = False

        model = tf.keras.Sequential()
        model.add(generator)
        model.add(discriminator)

        model.layers[0]._name = "WLencoder"
        model.layers[1]._name = "Discriminator"
        model.layers[1].trainable = False
        return model

    for cnt in range(epochs):
        start = time.time()
        ## train discriminator
        random_index = np.random.randint(0, train_count-batch)
        painting_images = X_train[random_index:random_index+batch]
        ground = y_train[random_index:random_index+batch]
        try:
            sample = tf.keras.models.load_model("./saved_model/temp.h5",
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
                                                                "color_loss": color_loss}
                                                )
            #sample.summary()
            try:
                sample.load_weights("./checkpoints/temp.h5")
            except:
                pass
        except:
            sample = gan_build(sample_w, sample_d)
            try:
                sample.load_weights("./checkpoints/temp.h5")
            except:
                pass

        generate = Model(inputs=sample.get_layer(name="WLencoder").input,
                         outputs=sample.get_layer(name="WLencoder").output)
        #generate.name="WLencoder"
        #generate.summary()
        syntetic_images = generate.predict(painting_images)
        #for i in range(border):
        #    syntetic_images[:, i, :, :] = painting_images[:, i, :, :]
        #    syntetic_images[:, i + border + masked, :, :] = painting_images[:, i + border + masked, :, :]
        #for j in range(border):
        #    syntetic_images[:, :, j, :] = painting_images[:, :, j, :]
        #    syntetic_images[:, :, j + border + masked, :] = painting_images[:, :, j + border + masked, :]

        x_combined_batch = np.concatenate((ground, syntetic_images))
        y_combined_batch = np.concatenate((np.ones((batch, 1)), np.zeros((batch, 1))))

        discriminate = Model(inputs=sample.get_layer(name="Discriminator").input,
                             outputs=sample.get_layer(name="Discriminator").output)
        discriminate.trainable = True
        #discriminate.name="Discriminator"
        discriminate.compile(optimizer=RMSprop(learning_rate=lr), loss=cross_entropy)

        d_loss = discriminate.train_on_batch(x_combined_batch, y_combined_batch)
        discriminate.save_weights("./saved_model/temp_d.h5")

        # train generator
        generate.compile(optimizer=RMSprop(learning_rate=lr), loss=Total_loss)
        g_loss = generate.train_on_batch(painting_images, ground)

        y_mislabled = np.ones((batch, 1))

        sample = gan_build(generate, discriminate)
        #sample.layers[0]._name = "WLencoder"
        #sample.layers[1]._name = "Discriminator"
        #sample.summary()
        sample.compile(optimizer=RMSprop(learning_rate=lr), loss=cross_entropy)
        a_loss = sample.train_on_batch(painting_images, y_mislabled)

        end = time.time()
        print('epoch: {}, [Discriminator :: d_loss: {}], [ Generator :: loss: {}], loss: {}, Time_used: {}'.format(cnt, d_loss, g_loss, a_loss, end-start))

        sample.save("./saved_model/temp.h5")
        sample.save("./saved_model/sample27.h5")
        sample.save_weights("./checkpoints/temp.h5")
        #sample.save_weights("./checkpoints/sample/sample27.h5")


train(images, grd, epochs=epochs, batch=batch_size)
#sample.save("./saved_model/sample27.h5")
#sample.save_weights("./checkpoint/sample/sample27")
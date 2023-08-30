import os
import numpy as np
from MaskGenerator import MaskGenerator
from PIL import Image


def find_path(cfg_name, shuffle=False):
    pth = []
    # cls = []
    with open(cfg_name, 'r') as f:
        text = f.read()
        text = text.split("\n")
        if shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(np.array(text))
        for p in text:
            temp = p.split(" ")
            pth.append(temp[0])
            # cls.append(temp[1])

    return pth

def find_path_fold(fold_name, shuffle=False):
    folds = os.listdir(fold_name)
    pths = []
    for fold in folds:
        tmp_pth = os.path.join(fold_name, fold)
        target = os.listdir(tmp_pth)
        for data in target:
            data_pth = os.path.join(tmp_pth, data)
            pths.append(data_pth)
    if shuffle == True:
        rng = np.random.default_rng()
        rng.shuffle(np.array(pths))

    return pths

def square_mask(image, images_shape=256, mask_size=64):
    image = image.reshape(1, images_shape, images_shape, 3)
    margin = int((images_shape / 2) - (mask_size / 2))
    for i in range(mask_size):
        for j in range(mask_size):
            image[:, i+margin, j+margin, :] = 0

    image = image.reshape(1, images_shape, images_shape, 3)
    image = image.astype("float")

    return image


#test = np.ones((1, 256, 256, 3))
#print(square_mask(test)[:, 62, 64, :])


def load_images(fold, pth, images_shape=256, num=10000, mask_type="square", masked=64, samp=0, dataset="Place", slice=False):
    arr = np.zeros((num, images_shape, images_shape, 3))
    arr2 = np.zeros((num, images_shape, images_shape, 3))
    count = 0
    mask_generator = MaskGenerator(images_shape, images_shape, 3, rand_seed=42)
    for p in pth[samp:samp+num]:
        if dataset == "Place":
            temp = os.path.join(fold, p)
            if temp == p:
                temp = fold + p
        else:
            temp = p
        image = Image.open(temp).convert("RGB")
        #image = image.resize((images_shape, images_shape), Image.LANCZOS)
        if slice:
            image = image.crop((0, 0, images_shape, images_shape))
        else:
            image = image.resize((images_shape, images_shape))
        img = np.array(image)
        mask = mask_generator.sample()
        if masked:
            if mask_type == "square":
                img = square_mask(img, images_shape, masked)
            elif mask_type == "free":
                #mask = mask_generator.sample()
                #img = np.add(img, np.full((images_shape, images_shape, 3), 1e-8))
                img = np.multiply(img, mask)
        img = img.reshape(1, images_shape, images_shape, 3)
        mask = mask.reshape(1, images_shape, images_shape, 3)
        arr[count] = img
        arr2[count] = mask
        count += 1

        print("Preprocessing: NO.", count)

    return arr, arr2


def Place_get_data(cfg_name, fold_name, image_shape=256, num=10000, mask_type="square", masked=96, shuffle=False, samp=0, slice=False):
    pth = find_path(cfg_name, shuffle=shuffle)
    X, X_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=masked, samp=samp, slice=slice)
    y, y_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=False, samp=samp, slice=slice)

    return X, X_mask, y, y_mask

def Place_get_slice(fold_name, image_shape=256, num=10000, mask_type="square", masked=96, shuffle=False, samp=0, slice=True):
    pth = find_path_fold(fold_name, shuffle=shuffle)
    X, X_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=masked, samp=samp, dataset="celeb", slice=slice)
    y, y_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=False, samp=samp, dataset="celeb", slice=slice)

    return X, X_mask, y, y_mask

def Celeba_HQ_get_data(fold_name, image_shape=256, num=10000, mask_type="square", masked=96, shuffle=False, samp=0, slice=False):
    pth = find_path_fold(fold_name, shuffle=shuffle)
    X, X_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=masked, samp=samp, dataset="celeb", slice=slice)
    y, y_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=False, samp=samp, dataset="celeb", slice=slice)

    return X, X_mask, y, y_mask

def Youtube_vos_get_data(fold_name, image_shape=256, num=10000, mask_type="square", masked=96, shuffle=False, samp=0, slice=False):
    pth = find_path_fold(fold_name, shuffle=shuffle)
    X, X_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=masked, samp=samp, dataset="celeb", slice=slice)
    y, y_mask = load_images(fold_name, pth, images_shape=image_shape, num=num, mask_type=mask_type, masked=False, samp=samp, dataset="celeb", slice=slice)

    return X, X_mask, y, y_mask

#images, images_mask, grd, grd_mask = Place2_get_data("./dataset/Place/places365_train_standard.txt", "./dataset/Place/data_256",
#                                                     image_shape=512, num=100, mask_type="free", masked=96, shuffle=True, samp=0)
#images, images_mask, grd, grd_mask = Celeba_HQ_get_data("./dataset/Celeba_HQ/train", image_shape=512, num=100, mask_type="free", masked=96, shuffle=True, samp=0)
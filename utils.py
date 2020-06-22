import PIL
import cv2
import numpy as np 
import h5py
import glob
import os
import math
from math import log10, sqrt


def PSNR(target, ref):
    target_img = np.array(target.astype(dtype = np.uint8))
    ref_img = np.array(ref.astype(dtype = np.uint8))

    mse = np.mean((ref_img - target_img) ** 2)
    if(mse == 0):
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel/sqrt(mse))
    return psnr


def imread(path):
    img = cv2.imread(path)
    return img


def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image)


def checkimage(image):
    cv2.imshow('test', image)
    cv2.waitKey(0)


def makenoisy(img, noise_level):
    noise = np.random.normal(0, noise_level, img.size)
    noise = np.reshape(noise, [img.shape[0], img.shape[1], img.shape[2]])
    noise = noise.astype('uint8')
    noisy_img = cv2.add(img, noise)
    return noisy_img


# def get_random_crop(input_, label_, config):

#     crop_height = config.image_size
#     crop_width = config.image_size

#     max_x = input_.shape[1] - crop_width
#     max_y = input_.shape[0] - crop_height

#     x = np.random.randint(0, max_x)
#     y = np.random.randint(0, max_y)

#     crop_input = input_[y: y + crop_height, x: x + crop_width]
#     crop_input = crop_input.reshape([config.image_size, config.image_size, config.c_dim])
#     crop_input = crop_input / 255.0

#     crop_label = label_[y: y + crop_height, x: x + crop_width]
#     crop_label = crop_label.reshape([config.image_size, config.image_size, config.c_dim])
#     crop_label = crop_label / 255.0


#     return crop_input, crop_label


# def preprocess(path, config):
#     noise_level = config.noise_level
    
#     img = imread(path)
    
#     input_ = makenoisy(img, noise_level)
#     label_ = img

#     # input_ = input_[:,:,::-1] # [::-1] means reverse ordering
#     # label_ = label_[:,:,::-1]
#     return input_, label_


# def prepare_data(config):
#     if config.is_train:
#         data_dir = os.path.join(os.path.join(os.getcwd(), 'Data'), config.train_set)
#         data_jpg = glob.glob(os.path.join(data_dir, '*.jpg'))
#         data_png = glob.glob(os.path.join(data_dir, '*.png'))
#         data = data_jpg + data_png
    
#     elif config.is_eval:
#         # data_dir = os.path.join(os.path.join(os.getcwd(), 'Data'), config.eval_set)
#         data_dir = './VOCdevkit/VOC2007/JPEGImages'
#         # data_dir = './Data/Train'
#         data = glob.glob(os.path.join(data_dir, '*.jpg'))
    
#     else:
#         if config.test_image != '':
#             data = [os.path.join(os.getcwd(), config.test_image)]
                
#         else: 
#             data_dir = os.path.join(os.path.join(os.getcwd(), 'Data'), config.test_set)
#             data = glob.glob(os.path.join(data_dir,'*.jpg'))

#     return data


# def make_data_h5(input_, label_, config, times):
#     if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
#         os.makedirs(os.join.path(os.getcwd(), config.checkpoint_dir))
    
#     if config.is_train:
#         save_path = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'train_%dsigma.h5' % config.noise_level)
#     else:
#         save_path = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'eval_%dsigma.h5' % config.noise_level)
    
#     if times == 0:
#         if os.path.exists(save_path):
#             print('\n%shave existed!!\n' % save_path)
#             return False
#         else:
#             hf = h5py.File(save_path, mode='w')

#             if config.is_train:
#                 input_h5 = hf.create_dataset('input', (1, config.image_size, config.image_size, config.c_dim), 
#                                             maxshape= (None, config.image_size, config.image_size, config.c_dim),
#                                             chunks= (1, config.image_size, config.image_size, config.c_dim), dtype= 'float32')
#                 label_h5 = hf.create_dataset('label', (1, config.image_size, config.image_size, config.c_dim), 
#                                             maxshape= (None, config.image_size, config.image_size, config.c_dim),
#                                             chunks= (1, config.image_size, config.image_size, config.c_dim), dtype= 'float32')
#             else:
#                 input_h5 = hf.create_dataset('input', (1, input_.shape[0], input_.shape[1], input_.shape[2]), 
#                                             maxshape= (None, input_.shape[0], input_.shape[1], input_.shape[2]),
#                                             chunks= (1, input_.shape[0], input_.shape[1], input_.shape[2]), dtype= 'float32')
#                 label_h5 = hf.create_dataset('label', (1, input_.shape[0], input_.shape[1], input_.shape[2]), 
#                                             maxshape= (None, input_.shape[0], input_.shape[1], input_.shape[2]),
#                                             chunks= (1, input_.shape[0], input_.shape[1], input_.shape[2]), dtype= 'float32')
#     else:
#         hf = h5py.File(save_path, 'a')
#         input_h5 = hf['input']
#         label_h5 = hf['label']

#     if config.is_train:
#         input_h5.resize([times+1, config.image_size, config.image_size, config.c_dim])
#         input_h5[times : times+1] = input_[np.newaxis,:]
#         label_h5.resize([times+1, config.image_size, config.image_size, config.c_dim])
#         label_h5[times : times+1] = label_[np.newaxis,:]
#     else:
#         input_h5.resize([times+1, input_.shape[0], input_.shape[1], input_.shape[2]])
#         input_h5[times : times+1] = input_[np.newaxis,:]
#         label_h5.resize([times+1, input_.shape[0], input_.shape[1], input_.shape[2]])
#         label_h5[times : times+1] = label_[np.newaxis,:]

#     hf.close()
#     return True



# def make_patched_data(data, config):
#     times = 0
#     for i in range(len(data)):
#         input_, label_ = preprocess(data[i], config)

#         if len(input_.shape) == 3:
#             h, w, c = input_.shape
#         else:
#             h, w = input_.shape

#         for patch in range(16):
#             crop_input, crop_label = get_random_crop(input_, label_, config)
#             save_flag = make_data_h5(crop_input, crop_label, config, times)
#             if not save_flag:
#                 return
#             times += 1

#         print('Processed training image: [%d / %d]' % (i+1, len(data)))


# def input_setup(config):
#     data = prepare_data(config)
#     make_patched_data(data, config)
#     return data


# def augmentation(batch, random):
#     if random[0] < 0.3:
#         batch_flip = np.flip(batch, 1)
#     elif random[0] > 0.7:
#         batch_flip = np.flip(batch, 2)
#     else:
#         batch_flip = batch
    
#     if random[1] < 0.5:
#         batch_rot = np.rot90(batch_flip, 1, [1,2])
#     else:
#         batch_rot = batch_flip
    
#     return batch_rot


# def get_data_dir(checkpoint_dir, is_train, noise_level):
#     if(is_train):
#         return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'train_%dsigma.h5' % noise_level)
#     else:
#         return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'eval_%dsigma.h5' % noise_level)


# def get_num_data(path):
#     with h5py.File(path, 'r') as hf:
#         input_ = hf['input']
#         return input_.shape[0]


# def get_batch(path, num_data, batch_size):
#     with h5py.File(path, 'r') as h5:
#         input_ = h5['input']
#         label_ = h5['label']

#         random_batch = np.random.rand(batch_size) * (num_data - 1)
#         batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
#         batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
#         for i in range(batch_size):
#             batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
#             batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

#         random_aug = np.random.rand(2)
#         batch_images = augmentation(batch_images, random_aug)
#         batch_labels = augmentation(batch_labels, random_aug)
#         return batch_images, batch_labels



# def get_image(path, config):
    
#     image, label = preprocess(path, config)
#     image = image[np.newaxis, :]
#     label = label[np.newaxis, :] 
#     return image, label


import tensorflow as tf
import numpy as np 
import time
import os

from utils import (
    input_setup,
    get_data_dir,
    get_num_data,
    get_batch,
    get_image,
    checkimage,
    imsave,
    imread,
    prepare_data,
    PSNR
)

class RDN(object):

    def __init__(self, sess, is_train, is_eval, image_size, noise_level, 
                c_dim, batch_size, D, C, G, G0, kernel_size):

            self.sess = sess
            self.is_train = is_train
            self.is_eval = is_eval
            self.image_size = image_size
            self.noise_level = noise_level
            self.c_dim = c_dim
            self.batch_size = batch_size
            self.D = D
            self.C = C
            self.G = G
            self.G0 = G0
            self.kernel_size = kernel_size

    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1' : tf.Variable(tf.random.normal([ks, ks, self.c_dim, G0], stddev=0.01), name = 'w_S_1'),
            'w_S_2' : tf.Variable(tf.random.normal([ks, ks, G0, G], stddev=0.01), name = 'w_S_2')
        }

        biasesS = {
            'b_S_1' : tf.Variable(tf.zeros([G0], name = 'b_S_1')),
            'b_S_2' : tf.Variable(tf.zeros([G], name = 'b_S_2'))
        }

        return weightsS, biasesS
    
    def RDBParams(self):
        weightsR = {}
        biasesR = {}

        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size

        for i in range(1, D+1):
            for j in range(1, C+1):
                weightsR.update({'w_R_%d_%d' % (i,j): tf.Variable(tf.random_normal([ks, ks, G *j, G], stddev=0.01), name= 'w_R_%d_%d' % (i,j))})
                biasesR.update({'b_R_%d_%d' % (i,j): tf.Variable(tf.zeros([G], name= 'w_R_%d_%d' % (i,j)))})
            weightsR.update({'w_R_%d_%d' % (i,C+1): tf.Variable(tf.random_normal([ks, ks, G * (C+1), G], stddev=0.01), name= 'w_R_%d_%d' % (i,C+1))})
            biasesR.update({'b_R_%d_%d' % (i,C+1): tf.Variable(tf.zeros([G], name= 'w_R_%d_%d' % (i,C+1)))})

        return weightsR, biasesR

    def DFFParams(self):

        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        ks = self.kernel_size

        weightsD = {
            'w_D_1' : tf.Variable(tf.random_normal([1, 1, G * D, G0], stddev= 0.01), name='w_D_1'),
            'w_D_2' : tf.Variable(tf.random_normal([ks, ks, G0, G0], stddev= 0.01), name='w_D_2')
        }

        biasesD= {
            'b_D_1' : tf.Variable(tf.zeros([G0], name='b_D_1')),
            'b_D_2' : tf.Variable(tf.zeros([G0], name='b_D_2'))
        }
        
        return weightsD, biasesD

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer

        for i in range(1, self.D + 1):
            x = rdb_in
            for j in range(1, self.C + 1):
                tmp = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i,j)], strides=[1,1,1,1], padding='SAME') + self.biasesR['b_R_%d_%d' % (i,j)]
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x,tmp], axis=3)
                
            x = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, self.C+1)], strides=[1,1,1,1], padding='SAME') + self.biasesR['b_R_%d_%d' % (i, self.C+1)]
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)
    
    def model(self):
        F_1 = tf.nn.conv2d(self.images, self.weightsS['w_S_1'], strides= [1,1,1,1], padding='SAME') + self.biasesS['b_S_1']

        F0 = tf.nn.conv2d(F_1, self.weightsS['w_S_2'], strides= [1,1,1,1], padding='SAME') + self.biasesS['b_S_2']

        FD = self.RDBs(F0)

        FGF1 = tf.nn.conv2d(FD, self.weightsD['w_D_1'], strides= [1,1,1,1], padding='SAME') + self.biasesD['b_D_1']

        FGF2 = tf.nn.conv2d(FGF1, self.weightsD['w_D_2'], strides= [1,1,1,1], padding='SAME') + self.biasesD['b_D_2']

        FDF = tf.add(FGF2, F_1)

        IHR = tf.nn.conv2d(FDF, self.weight_final, strides= [1,1,1,1], padding='SAME') + self.bias_final

        IHQ = tf.add(IHR, self.images)
        
        return IHQ

    def build_model(self,images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name = 'images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name = 'labels')

        self.weightsS, self.biasesS = self.SFEParams()
        self.weightsR, self.biasesR = self.RDBParams()
        self.weightsD, self.biasesD = self.DFFParams()
        self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.G0, self.c_dim], stddev= np.sqrt(2.0/9/3)), name='w_f')
        self.bias_final = tf.Variable(tf.zeros([self.c_dim], name='b_f'))

        self.preds = self.model()
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.preds))
        self.summary = tf.summary.scalar('loss', self.loss)
        self.model_name = '%s_%s_%s_%s_sig_%s' % ('rdn', self.D, self.C, self.G, self.noise_level)
        self.saver = tf.train.Saver(max_to_keep = 10)


    def load(self, checkpoint_dir, restore=True):
        ckpt_dir = os.path.join(checkpoint_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt_path).split('-')[1])
            if restore:
                self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
                print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            if restore:
                print("\nEither checkpoint not available or Checkpoint Loading Failed! \n")

        return step


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "RDN.model"),
                        global_step=step)

    
    def train(self,config):
        print('\nPreparing Data....\n')
        data = input_setup(config)
        if len(data)==0:
            print('\nTraining data not found\n')
            return

        data_dir = get_data_dir(config.checkpoint_dir, config.is_train, config.noise_level)
        print((data_dir))
        num_data = get_num_data(data_dir)    
        print(num_data)
        num_batch = num_data // config.batch_size
        print(num_batch)

        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]
        self.build_model(images_shape, labels_shape)

        counter = self.load(config.checkpoint_dir, restore=False)
        epoch_start = int(counter / num_batch)
        batch_start = counter % num_batch

        global_step = tf.Variable(counter, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps * num_batch, config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        learning_step = optimizer.minimize(loss= self.loss, global_step = global_step)

        self.sess.run(tf.global_variables_initializer())
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(config.checkpoint_dir, self.model_name, 'log'), graph=self.sess.graph)

        self.load(config.checkpoint_dir, restore = True)
        print('\nStarting training...\n')
        logFile = open('train_log_file_sigma_%s.txt' % config.noise_level, 'w')
        for epoch in range(epoch_start, config.epochs):
            for idx in range(batch_start, num_batch):
                batch_images, batch_labels = get_batch(data_dir, num_data, config.batch_size)
                counter += 1

                _, err, lr = self.sess.run([learning_step, self.loss, learning_rate], feed_dict= {self.images: batch_images, self.labels: batch_labels})

                if counter % 10 == 0:
                    print('Epoch: [%4d] Batch: [%d/%d] loss: [%.8f] lr: [%5f] step: [%d]' % ((epoch+1), (idx+1), num_batch, err, lr, counter))
                    logFile.write('Epoch: [%4d] Batch: [%d/%d] loss: [%.8f] lr: [%5f] step: [%d]\n' % ((epoch+1), (idx+1), num_batch, err, lr, counter))
                if counter % 10000 == 0:
                    self.save(config.checkpoint_dir, counter)

                    summary_str = self.sess.run(merged_summary_op, feed_dict = {self.images: batch_images, self.labels: batch_labels})
                    summary_writer.add_summary(summary_str, counter)

                if counter > 0 and counter == config.epochs * num_batch:
                    self.save(config.checkpoint_dir,counter)
                    break

        logFile.close()
        summary_writer.close()

#     def eval(self, config):
#         print('\nPreparing Data..\n')
#         paths = prepare_data(config)
#         num_data = len(paths)
    
#         psnrFile = open('psnr_sigma_%s.txt'%config.noise_level, 'a')

#         avg_time = 0
#         avg_psnr = 0
#         print('\nNow evaluating the dataset\n')
#         for i in range(num_data):

#             input_, label_ = get_image(paths[i], config)

#             image_shape = input_.shape
#             label_shape = label_.shape

#             self.build_model(image_shape, label_shape)

#             self.sess.run(tf.global_variables_initializer())

#             self.load(config.checkpoint_dir, restore=True)

#             time_ = time.time()
#             result = self.sess.run([self.preds], feed_dict = {self.images: input_/255.0})
#             avg_time += time.time() - time_

#             self.sess.close()
#             tf.reset_default_graph()
#             self.sess = tf.Session()

#             img = np.squeeze(result) * 255
#             img = np.clip(img, 0, 255)
#             psnr = PSNR(img, label_)
#             avg_psnr += psnr

#             print('image: [%d/%d] time: [%.4f] psnr: [%.4f]' % (i+1, num_data, time.time()-time_, psnr))
#             psnrFile.write('image: [%d/%d] time: [%.4f] psnr: [%.4f]\n' % (i+1, num_data, time.time()-time_, psnr))

#             if not os.path.isdir(os.path.join(os.getcwd(), config.result_dir)):
#                 os.makedirs(os.path.join(os.getcwd(), config.result_dir))
            
#             filename = os.path.basename(paths[i])
#             imsave(img, path= config.result_dir + '/%d_sigma/JPEGImages/' % config.noise_level + filename)

#         print('\nAverage Time: %.4f' % (avg_time / num_data))
#         psnrFile.write('\nAverage Time: %.4f' % (avg_time / num_data))
#         print('\nAverage PSNR: %.4f' % (avg_psnr / num_data))
#         psnrFile.write('\nAverage PSNR: %.4f' % (avg_psnr / num_data))
#         psnrFile.close()
 


#     def test(self, config):
#         print('\nPreparing the data..\n')
#         paths = prepare_data(config)
#         num_data = len(paths)

#         avg_time = 0
#         print('\nInitiating the testing\n')
#         for i in range(num_data):
# #            input_ = imread(paths[i])
# #            input_ = input_[:,:,::-1]
#             input_, label_ = get_image(paths[i], config)
# #            input_ = input_[np.newaxis, :]

#             image_shape = input_.shape
#             label_shape = input_.shape
#             self.build_model(image_shape, label_shape)
#             tf.global_variables_initializer().run(session= self.sess)

#             self.load(config.checkpoint_dir, restore=True)

#             time_ = time.time()
#             result = self.sess.run([self.preds], feed_dict = {self.images: input_/255.0})
#             avg_time += time.time() - time_

#             self.sess.close()
#             tf.reset_default_graph()
#             self.sess = tf.Session()

#             img = np.squeeze(result) * 255
#             img = np.clip(img, 0, 255)
# #            img = img[:,:,::-1]

#             filename = os.path.basename(paths[i])
#             if not os.path.isdir(os.path.join(os.getcwd(), config.result_dir)):
#                 os.makedirs(os.path.join(os.getcwd(), config.result_dir))
#             imsave(img, path= config.result_dir + '/' +filename)
#             imsave(input_[0, :], path= config.result_dir + '/noisy_%s' % filename)
        

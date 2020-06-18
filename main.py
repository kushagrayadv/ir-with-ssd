import tensorflow as tf 
from rdn_model import RDN 
import ssd_network
from datasets import pascalvoc_to_tfrecords
import os
import glob

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_train', False, 'if the training')
flags.DEFINE_boolean('is_eval', False, 'if the evaluation')
flags.DEFINE_integer('image_size', 48, 'the size of the input image')
flags.DEFINE_integer('noise_level', 11, 'the size of the input image')
flags.DEFINE_integer('c_dim', 3, 'number of channels')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'the size of the input image-batch')
flags.DEFINE_float('learning_rate', 1e-4, 'the learning rate')
flags.DEFINE_float('lr_decay_steps', 10, 'steps of learning rate decay')
flags.DEFINE_float('lr_decay_rate', 0.5, 'rate of the learning rate decay')
flags.DEFINE_string('test_image', '', 'the test image')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'name of the checkpoint directory')
flags.DEFINE_string('result_dir', 'result', 'name of the result directory')
flags.DEFINE_string('train_set', 'Train', 'name of the training set')
flags.DEFINE_string('eval_set', 'Eval', 'name of the evaluation set')
flags.DEFINE_string('test_set', 'Test', 'name of the test set')
flags.DEFINE_integer('D', 10, 'D')
flags.DEFINE_integer('C', 5, 'C')
flags.DEFINE_integer('G', 64, 'G')
flags.DEFINE_integer('G0', 64, 'G0')
flags.DEFINE_integer('kernel_size', 3, 'the size of the kernel')


def main(_):
    
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(log_device_placement = False, gpu_options = gpu_options)


    sess = tf.Session(config = config)
    rdn = RDN(sess, 
            is_train = FLAGS.is_train,
            is_eval = FLAGS.is_eval,
            image_size = FLAGS.image_size,
            noise_level = FLAGS.noise_level,
            c_dim = FLAGS.c_dim,
            batch_size = FLAGS.batch_size,
            D = FLAGS.D,
            C = FLAGS.C,
            G = FLAGS.G,
            G0 = FLAGS.G0,
            kernel_size = FLAGS.kernel_size)

    if(rdn.is_train):
        rdn.train(FLAGS)
    
    else:

        if rdn.is_eval:
            rdn.eval(FLAGS)
            dataset_dir = os.path.join(os.path.join(os.getcwd(), FLAGS.result_dir), '%s_sigma/'%FLAGS.noise_level)
            output_eval_path = os.path.join(os.path.join(os.getcwd(), 'Data'), FLAGS.eval_set)
            output_eval_path = output_eval_path + '/sigma_%s'%FLAGS.noise_level
            if not os.path.exists(output_eval_path):
                os.makedirs(output_eval_path)
            pascalvoc_to_tfrecords.run(dataset_dir, output_eval_path, 'voc_2007_test')
            
            eval_logdir = os.path.join(os.path.join(os.getcwd(), 'logs'),'%d_sigma'%FLAGS.noise_level)
            output_eval_path = output_eval_path + '/'
            ssd_network.ssd_eval('pascalvoc_2007', 
                                 dataset_dir= output_eval_path, 
                                 batch_size= FLAGS.batch_size,
                                 eval_dir = eval_logdir)

        else:
            # Testing the whole model
            # testing is first done on RDN network
#            rdn.test(FLAGS)
            # Then, testing the images on SSD network for object detection
#            test_path = os.path.join(os.path.join(os.getcwd(), 'Test'), FLAGS.test_set)
            test_path = os.path.join(os.getcwd(), FLAGS.result_dir, 'test')
            ssd_network.ssd_test(test_path)


if __name__ == "__main__":
    tf.app.run()   


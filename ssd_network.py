import tensorflow as tf 
import numpy as np 
import os
import cv2
import math
import time
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing
import visualization
from datasets import dataset_factory
import tf_extended as tfe 

slim = tf.contrib.slim


# SSD 300 MODEL EVALUATION PARAMS

net_shape = (300, 300)

data_format = 'NHWC'

# Define the model checkpoint path
ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'



# Function to reshape 1D list to 2D
def reshape_list(l, shape=None):
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


def flatten(x): 
         result = [] 
         for el in x: 
              if isinstance(el, tuple): 
                    result.extend(flatten(el))
              else: 
                    result.append(el) 
         return result



# Function for evaluating the network
def ssd_eval(dataset_name, dataset_dir, batch_size, eval_dir):
    
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():

        tf_global_step = slim.get_or_create_global_step()
        
        # Dataset + SSD Model + Pre-processing
        dataset = dataset_factory.get_dataset(dataset_name, 'test', dataset_dir)
        
        ssd_net = ssd_vgg_300.SSDNet()
        ssd_shape = net_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Create a dataset provider and batches
        with tf.device('/cpu:0'):
            with tf.name_scope(dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    common_queue_capacity = 2 * batch_size,
                    common_queue_min = batch_size,
                    shuffle = False
                )

            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape','object/label', 'object/bbox'])
            [gdifficults] = provider.get(['object/difficult'])

            image, glabels, gbboxes, gbbox_img = ssd_vgg_preprocessing.preprocess_for_eval(image, glabels, gbboxes, ssd_shape,
                                                                    data_format = data_format, 
                                                                    resize = ssd_vgg_preprocessing.Resize.WARP_RESIZE)

            gclasses, glocalizations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] * 5 + [len(ssd_anchors)] * 3

            # Evaluation Batch
            r = tf.train.batch(reshape_list([image, glabels, gbboxes, gdifficults, gbbox_img, gclasses, glocalizations, gscores]),
                                batch_size = batch_size, 
                                num_threads = 1,
                                capacity = 5 * batch_size, 
                                dynamic_pad = True)
            (b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses, b_glocalizations,
                b_gscores) = reshape_list(r, batch_shape)

        # SSD network + output decoding
        arg_scope = ssd_net.arg_scope(data_format= data_format)
        with slim.arg_scope(arg_scope):
            predictions, localizations, logits, _ = ssd_net.net(b_image, is_training=False)
            
        ssd_net.losses(logits, localizations, b_gclasses, b_glocalizations, b_gscores)

        with tf.device('/device:CPU:0'):
            localizations = ssd_net.bboxes_decode(localizations, ssd_anchors)
            rscores, rbboxes = ssd_net.detected_bboxes(predictions, localizations,
                                                        select_threshold=0.01,
                                                        nms_threshold=0.45,
                                                        clipping_bbox=None,
                                                        top_k=400,
                                                        keep_top_k=200)
            
            num_gbboxes, tp, fp, rscores = tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                                                    b_glabels, b_gbboxes, b_gdifficults, 
                                                                    matching_threshold= 0.5)
            
        variables_to_restore = slim.get_variables_to_restore()

   
        with tf.device('/device:CPU:0'):

            dict_metrics = {}
            
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            
            for name, metric in dict_metrics.items():
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c], tp_fp_metric[1][c])

            aps_VOC07 = {}
            aps_voc12 = {}
            
            for c in tp_fp_metric[0].keys():
                # precision and recall values
                pre, rec = tfe.precision_recall(*tp_fp_metric[0][c])
                
                # average precision VOC07
                v = tfe.average_precision_voc07(pre,rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_VOC07[c] = v

                # Average precision VOC12.
                v = tfe.average_precision_voc12(pre, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v
            
            # Mean average Precision VOC07
            summary_name = 'AP_VOC07/mAP'
            mAP = tf.add_n(list(aps_VOC07.values()))/len(aps_VOC07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

#         # Evaluation Loop

#         gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
#         config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)


#         num_batches = math.ceil(dataset.num_samples / float(batch_size))
#         tf.logging.info('Evaluating %s' % ckpt_filename)
#         start = time.time()
#         slim.evaluation.evaluate_once(master= '', 
#                                       checkpoint_path = ckpt_filename,
#                                       logdir= eval_dir, 
#                                       num_evals= num_batches,
#                                       eval_op= flatten(list(names_to_updates.values())),
#                                       variables_to_restore= variables_to_restore,
#                                       session_config = config)
#         # log time spent
#         elapsed = time.time() - start
#         print('Time Spent: %.3f' % elapsed)
#         print('Time Spent per batch: %.3f seconds' % (elapsed/num_batches))



# def ssd_test(path):

#     # Input Placeholder
#     img_input = tf.placeholder(tf.uint8, shape= (None, None, 3))

#     # Evaluation pre-processing: resize to ssd net shape
#     image_pre, labels_pre, bbox_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input,
#                                                     None, None, net_shape, data_format,
#                                                     resize= ssd_vgg_preprocessing.Resize.WARP_RESIZE)
#     image_4d = tf.expand_dims(image_pre, axis=0)

#     # Define the SSD model
#     reuse = True if 'ssd_net' in locals() else None
#     ssd_net = ssd_vgg_300.SSDNet()
#     with slim.arg_scope(ssd_net.arg_scope(data_format= data_format)):
#         predictions, localizations, _, _ = ssd_net.net(image_4d, is_training= False, reuse = reuse)


#     # SSD default anchor boxes
#     ssd_anchors = ssd_net.anchors(net_shape)

#     # Main image processing pipeline

#     # Tensorflow Session: grow memeory when needed, do not allow full GPU usage
#     gpu_options = tf.GPUOptions(allow_growth = True)
#     config = tf.ConfigProto(log_device_placement = False, gpu_options = gpu_options)
    
#     isess = tf.InteractiveSession(config = config)

#     # Restore the SSD model
#     isess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(isess, ckpt_filename)

#     # Run the SSD network
#     def post_process(img, select_thresh=0.5, nms_thresh=0.45):
#         rimg, rpredictions, rlocalizations, rbbox_img = isess.run([image_4d, predictions, localizations, bbox_img],
#                                                             feed_dict= {img_input: img})
        
#         # get the classes and bboxes from the output
#         rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalizations,
#                                                             ssd_anchors, select_threshold=select_thresh,
#                                                             img_shape = net_shape, num_classes = 21,
#                                                             decode = True)
        
#         rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
#         rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k = 400)
#         rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold = nms_thresh)

#         # Resize the bboxes to the original image sizes, but useless for Resize.WARP
#         rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

#         return rclasses, rscores, rbboxes
    
    
#     imgs = os.listdir(path)
#     for i in range(len(imgs)):
#         img_path = os.path.join(path, imgs[i])
#         img = mpimg.imread(img_path)
#         rclasses, rscores, rbboxes = post_process(img)
#         visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

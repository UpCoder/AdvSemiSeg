import tensorflow as tf
import numpy as np
from model.deeplab_tf import ResNet, Bottleneck


def calc_seg_loss(pred, gt):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pred))


def train():
    input_size = 400
    learning_rate_init = 2.5e-4
    momentum = 0.9
    WEIGHT_DECAY = 0.0005
    pre_epoch_num = 100
    epoch_num = 2000

    # build for dataset
    from dataset.LiverDataset.liver_dataset import LiverDataset
    user_name = 'give'
    validation_interval = 800
    max_steps = 1000000000
    batch_size = 1
    n_neighboringslices = 1
    input_size = 400
    output_size = 400
    slice_type = 'axial'
    oversample = False
    # reset_counter = args.reset_counter
    label_of_interest = 1
    label_required = 0
    magic_number = 26.91
    max_slice_tries_val = 0
    max_slice_tries_train = 2
    fuse_labels = True
    apply_crop = False

    train_data_dir = "/home/" + user_name + "/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2"
    test_data_dir = "/home/" + user_name + "/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1"
    train_dataset = LiverDataset(
        data_dir=train_data_dir, slice_type=slice_type, n_neighboringslices=n_neighboringslices, input_size=input_size,
        oversample=oversample, label_of_interest=label_of_interest, label_required=label_required,
        max_slice_tries=max_slice_tries_train, fuse_labels=fuse_labels, apply_crop=apply_crop,
        interval=validation_interval, is_training=True, batch_size=batch_size, data_augmentation=True
    )
    val_dataset = LiverDataset(
        data_dir=test_data_dir, slice_type=slice_type, n_neighboringslices=n_neighboringslices, input_size=input_size,
        oversample=oversample, label_of_interest=label_of_interest, label_required=label_required,
        max_slice_tries=max_slice_tries_val, fuse_labels=fuse_labels, apply_crop=apply_crop,
        interval=validation_interval, is_training=False, batch_size=batch_size
    )

    # build for tensorflow of logits
    input_tensor = tf.placeholder(tf.float32, [None, input_size, input_size, 1], name='x_input')
    gt_tensor = tf.placeholder(tf.int32, [None, input_size, input_size], name='label_input')
    is_training_tensor = tf.placeholder(tf.bool, [], name='is_training')
    resnet = ResNet(input_tensor, Bottleneck, [3, 4, 23, 3], num_classes=2, is_training=is_training_tensor)
    resnet_output = resnet.out
    pixelwise_prediction = tf.image.resize_images(resnet_output, [input_size, input_size], align_corners=True)

    # build for tensorflow of loss and trian_op
    global_step_op = tf.train.get_or_create_global_step()
    loss_op = calc_seg_loss(pixelwise_prediction, gt_tensor)
    learning_rate_op = tf.train.exponential_decay(learning_rate_init, global_step_op, pre_epoch_num, decay_rate=0.9, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_op, momentum=momentum)
    train_op = optimizer.minimize(loss_op)

    # build for tensorflow of measurement
    pixelwise_prediction_argmax = tf.argmax(pixelwise_prediction, axis=-1)
    num_prediction_op = tf.reduce_sum(tf.cast(tf.greater_equal(pixelwise_prediction_argmax, 1), tf.uint8))
    num_ground_truth_op = tf.reduce_sum(tf.cast(tf.greater_equal(gt_tensor, 1), tf.uint8))
    num_intersection_op = tf.reduce_sum(
        tf.cast(tf.logical_and(
            tf.greater_equal(pixelwise_prediction_argmax, 1),
            tf.greater_equal(gt_tensor, 1)
        ), tf.uint8)
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_idx in range(epoch_num):
            total_num_prediction = 0
            total_num_groun_truth = 0
            total_num_intersection = 0
            loss_list = []
            for step_idx in range(pre_epoch_num):
                batch_image, batch_mask = train_dataset.get_next_batch()
                feed_dict = {
                    input_tensor: batch_image,
                    gt_tensor: batch_mask,
                    is_training_tensor: True
                }

                _, loss_value, num_prediction, num_ground_truth, num_intersection = sess.run(
                    [train_op, loss_op, num_prediction_op, num_ground_truth_op, num_intersection_op],
                    feed_dict=feed_dict)
                total_num_groun_truth += num_ground_truth
                total_num_intersection += num_intersection
                total_num_prediction += num_prediction
                loss_list.append(loss_value)
            dice = (2. * total_num_intersection + 1e-7) / (total_num_prediction + total_num_groun_truth + 1e-7)

            print('epoch: %d / %d, loss: %.4f, prediction: %d, ground truth: %d, intersection: %d. dice: %.4f' %
                  (epoch_idx, epoch_num, np.mean(loss_list), total_num_prediction, total_num_groun_truth,
                   total_num_intersection, dice))
            loss_list = []


if __name__ == '__main__':
    train()

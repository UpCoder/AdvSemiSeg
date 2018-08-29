import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import pickle
from packaging import version
# from model.deeplab_UNet_BiCLSTM import Res_Deeplab
from model.deeplab_BiCLSTM import Res_Deeplab
from glob import glob
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet
import tensorflow as tf


import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 100
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 20000000
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
# RESTORE_FROM = '/home/ld/github/AdvSemiSeg/pretrained/resnet101COCO-41f33a49.pth'
# RESTORE_FROM = '/home/give/github/AdvSemiSeg/snapshots_Unet_BiCLSTM/Liver261.pth'
# iter_start = 261
iter_start = 0
RESTORE_FROM = None
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = './snapshots_BiCLSTM/'
SUMMARY_DIR = './log_BiCLSTM/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA = 0.5

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True

UPDATE_TENSORBOARD_INTERVAL = 100
VAL_EXECUTE_TIMES = 1
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label


def save_model(model, snapshot_dir, perfix_name, i_iter, max_save_num=10):
    paths = list(glob(osp.join(snapshot_dir, perfix_name + '*.pth')))
    paths.sort()
    print(paths)
    if paths is not None and len(paths) >= max_save_num:
        os.remove(paths[0])
    torch.save(model.state_dict(), osp.join(snapshot_dir, perfix_name + str(i_iter) + '.pth'))


def main():
    # LD ADD start
    from dataset.LiverDataset.liver_dataset import LiverDataset
    user_name = 'give'
    validation_interval = 800
    max_steps = 1000000000
    batch_size = 1
    n_neighboringslices = 5
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
    # LD ADD end

    # LD build for summary
    training_summary = tf.summary.FileWriter(os.path.join(SUMMARY_DIR, 'train'))
    val_summary = tf.summary.FileWriter(os.path.join(SUMMARY_DIR, 'val'))
    dice_placeholder = tf.placeholder(tf.float32, [], name='dice')
    loss_placeholder = tf.placeholder(tf.float32, [], name='loss')
    tf.summary.scalar('dice', dice_placeholder)
    tf.summary.scalar('loss', loss_placeholder)
    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    perfix_name = 'Liver'
    h, w = map(int, args.input_size.split(','))
    # input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes, slice_num=n_neighboringslices)
    if RESTORE_FROM is not None:
        # load pretrained parameters
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        # only copy the params that exist in current model (caffe-like)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            print(name)
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
                print('copy {}'.format(name))
        model.load_state_dict(new_params)


    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # LD delete
    '''
    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda(args.gpu)
    '''

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # LD delete
    '''
    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()
    '''

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size, input_size), mode='bilinear')

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size, input_size), mode='bilinear')


    # labels for adversarial training
    pred_label = 0
    gt_label = 1
    loss_list = []
    dice_list = []
    for i_iter in range(iter_start, args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0
        num_prediction = 0
        num_ground_truth = 0
        num_intersection = 0


        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        # model.train(True)
        # LD delete
        '''
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
        '''
        for sub_i in range(args.iter_size):

            # LD delete
            '''
            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.next()

            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)
            '''
            batch_image, batch_label = train_dataset.get_next_batch()
            batch_image = np.transpose(batch_image, axes=(0, 3, 1, 2))
            # batch_image = np.concatenate([batch_image, batch_image, batch_image], axis=1)
            # print('Batch_images: ', np.shape(batch_image))

            batch_image_torch = torch.Tensor(batch_image)
            images = (batch_image_torch).cuda(args.gpu)

            # LD delete
            # ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))
            pred_ny = pred.data.cpu().numpy()
            pred_ny = np.transpose(pred_ny, axes=(0, 2, 3, 1))
            pred_label_ny = np.squeeze(np.argmax(pred_ny, axis=3))

            cur_prediction = np.sum(np.asarray(pred_label_ny, np.uint8))
            cur_grount_truth = np.sum(np.asarray(batch_label >= 1, np.uint8))
            cur_intersection = np.sum(np.asarray(np.logical_and(batch_label >= 1, pred_label_ny >= 1), np.uint8))
            cur_dice = (2 * cur_intersection + 1e-7) / (cur_prediction + cur_grount_truth + 1e-7)
            dice_list.append(cur_dice)

            num_prediction += np.sum(np.asarray(pred_label_ny, np.uint8))
            num_ground_truth += np.sum(np.asarray(batch_label >= 1, np.uint8))
            num_intersection += np.sum(np.asarray(np.logical_and(batch_label >= 1, pred_label_ny >= 1), np.uint8))
            # num_intersection += np.sum(np.asarray(batch_label >= 1, np.uint8) == np.asarray(pred_label_ny, np.uint8))

            loss_seg = loss_calc(pred, batch_label, args.gpu)

            # LD delete
            '''
            D_out = interp(model_D(F.softmax(pred)))

            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred
            '''
            loss = loss_seg
            # print('Loss is: ', loss)
            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            # print('Loss of numpy is: ', loss_seg.data.cpu().numpy())
            # print('Loss of numpy of zero is: ', loss_seg.data.cpu().numpy())
            loss_seg_value += loss_seg.data.cpu().numpy()
            loss_list.append(loss_seg_value)
            # loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()[0]/args.iter_size

        optimizer.step()
        # optimizer_D.step()
        dice = (2 * num_intersection + 1e-7) / (num_prediction + num_ground_truth + 1e-7)
        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value))
        print('dice: %.4f, num_prediction: %d, num_ground_truth: %d, num_intersection: %d' % (dice, num_prediction,
                                                                                              num_ground_truth,
                                                                                              num_intersection))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, perfix_name + str(args.num_steps)+'.pth'))
            # torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, perfix_name +str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            # torch.save(model.state_dict(), osp.join(args.snapshot_dir, perfix_name + str(i_iter)+'.pth'))
            save_model(model, args.snapshot_dir, perfix_name, i_iter, 2)
            # torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, perfix_name +str(i_iter)+'_D.pth'))


        # if i_iter % UPDATE_TENSORBOARD_INTERVAL and i_iter != 0:
        # update tensorboard
        feed_dict = {
            dice_placeholder: dice,
            loss_placeholder: np.mean(loss_list)
        }
        summery_value = sess.run(summary_op, feed_dict)
        training_summary.add_summary(summery_value, i_iter)
        training_summary.flush()
        loss_list = []
        dice_list = []
        # for validation
        # val_num_prediction = 0
        # val_num_ground_truth = 0
        # val_num_intersection = 0
        # loss_list = []
        # # model.train(False)
        # for idx in range(VAL_EXECUTE_TIMES):
        #     print(idx)
        #     batch_image, batch_label = val_dataset.get_next_batch()
        #     batch_image = np.transpose(batch_image, axes=(0, 3, 1, 2))
        #     # batch_image = np.concatenate([batch_image, batch_image, batch_image], axis=1)
        #     # print('Shape: ', np.shape(batch_image))
        #     # batch_image_torch = torch.Tensor(batch_image)
        #     images = Variable(batch_image, volatile=True).cuda()
        #
        #     # LD delete
        #     # ignore_mask = (labels.numpy() == 255)
        #     pred = interp(model(images))
        #     pred_ny = pred.data.cpu().numpy()
        #     pred_ny = np.transpose(pred_ny, axes=(0, 2, 3, 1))
        #     pred_label_ny = np.squeeze(np.argmax(pred_ny, axis=3))
        #     val_num_prediction += np.sum(np.asarray(pred_label_ny, np.uint8))
        #     val_num_ground_truth += np.sum(np.asarray(batch_label >= 1, np.uint8))
        #     val_num_intersection += np.sum(np.asarray(np.logical_and(batch_label >= 1, pred_label_ny >= 1), np.uint8))
        #
        #     loss_seg = loss_calc(pred, batch_label, args.gpu)
        #     loss_seg_value += loss_seg.data.cpu().numpy()
        #     loss_list.append(loss_seg_value)
        # dice = (2 * val_num_intersection + 1e-7) / (val_num_prediction + val_num_ground_truth + 1e-7)
        # feed_dict = {
        #     dice_placeholder: dice,
        #     loss_placeholder: np.mean(loss_list)
        # }
        # print('validation: dice:%.4f, loss: %.4f' % (dice, np.mean(loss_list)))
        # summery_value = sess.run(summary_op, feed_dict)
        # val_summary.add_summary(summery_value, i_iter)
        # val_summary.flush()
        # loss_list = []
        # print('\n')
    training_summary.close()
    val_summary.close()
    end = timeit.default_timer()
    print(end-start, 'seconds')


if __name__ == '__main__':
    print('Training UNet CLSTM')

    main()

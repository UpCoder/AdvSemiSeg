import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
from packaging import version
from dataset.LiverDataset.medicalImage import read_image_file, processing, preprocessing_agumentation
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplab import Res_Deeplab as original_model
from model.deeplab_UNet import Res_Deeplab as unet_model
from model.deeplab_UNet_CLSTM import Res_Deeplab as unet_clstm_model
from model.deeplab_UNet_BiCLSTM import Res_Deeplab as unet_biclstm_model
from dataset.voc_dataset import VOCDataSet

from PIL import Image
import utils.niifs as niifs
import utils.niiplot as niiplot
import utils.niismooth as niismooth
import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth'
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'


pretrianed_models_dict ={'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
                         'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
                         'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
                         'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))


    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


def execute_prediction_one_case(image_path, size_training, n_neighboringslices, batch_size=1):
    # about the model
    num_classes = 2
    cuda_gpu = 0
    original_size = 512
    model = Res_Deeplab(num_classes=num_classes)
    restore_path = '/home/ld/github/AdvSemiSeg/snapshots/Liver115.pth'

    saved_state_dict = torch.load(restore_path)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(cuda_gpu)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(original_size, original_size), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(original_size, original_size), mode='bilinear')

    def prediction_one_batch(model, batch_image, interp, cuda_gpu):
        batch_image = np.transpose(batch_image, axes=(0, 3, 1, 2))
        batch_image = np.concatenate([batch_image, batch_image, batch_image], axis=1)
        # print('Shape: ', np.shape(batch_image))
        batch_image_torch = torch.Tensor(batch_image)
        output = model(Variable(batch_image_torch, volatile=True).cuda(cuda_gpu))
        output = interp(output).cpu().data[0].numpy()
        return output


    image = read_image_file(image_path)
    original_image = np.copy(image)
    if image is None:
        print('The image is None!')
        assert False
    processed_image = processing(original_image, size_training)
    smoother = niismooth.Smoothutil()
    half_num_slice = n_neighboringslices // 2
    image_input = np.zeros(
        [1, size_training, size_training, n_neighboringslices], dtype=np.float32)

    for slice_idx in range(processed_image.shape[2]):
        for j in range(n_neighboringslices):
            cur_idx = slice_idx - half_num_slice + j
            if cur_idx < 0:
                cur_idx = 0
            if cur_idx >= processed_image.shape[2]:
                cur_idx = processed_image.shape[2] - 1

            image_input[0, :, :, j] = processed_image[:, :, cur_idx]
        pred = prediction_one_batch(model, image_input, interp, cuda_gpu)
        print(np.shape(pred))


def evaluate_measure_one_case(image_path, gt_path, restore_path, size_training, n_neighboringslices,
                              batch_size=1, flag='original'):
    optimizer = True
    # about the model
    num_classes = 2
    cuda_gpu = 0
    original_size = 512
    if flag == 'original':
        model = original_model(num_classes=num_classes)
        restore_path = '/home/give/github/AdvSemiSeg/snapshots/Liver1947.pth'
    if flag == 'unet':
        model = unet_model(num_classes=num_classes)
        restore_path = '/home/give/github/AdvSemiSeg/snapshots_Unet/Liver7555.pth'
    if flag == 'unet_clstm':
        model = unet_clstm_model(num_classes=num_classes)
        restore_path = '/home/give/github/AdvSemiSeg/snapshots_Unet_CLSTM/Liver4141.pth'
    if flag == 'unet_biclstm':
        model = unet_biclstm_model(num_classes=num_classes)
        restore_path = '/home/give/github/AdvSemiSeg/snapshots_Unet_BiCLSTM/Liver844.pth'
    restore_path = restore_path
    print('restore from ', restore_path)
    saved_state_dict = torch.load(restore_path)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(cuda_gpu)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(original_size, original_size), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(original_size, original_size), mode='bilinear')

    def prediction_one_batch(model, batch_image, interp, cuda_gpu):
        batch_image = np.transpose(batch_image, axes=(0, 3, 1, 2))
        # print('batch_image shape: ', np.shape(batch_image))
        if flag == 'unet' or flag == 'original':
            batch_image = np.concatenate([batch_image, batch_image, batch_image], axis=1)
        # print('Shape: ', np.shape(batch_image))
        batch_image_torch = torch.Tensor(batch_image)
        output = model(Variable(batch_image_torch, volatile=True).cuda(cuda_gpu))
        output = interp(output).cpu().data[0].numpy()
        return output

    image = read_image_file(image_path)
    gt_image = read_image_file(gt_path)

    original_image = np.copy(image)
    if image is None:
        print('The image is None!')
        assert False
    if flag == 'original':
        processed_image = processing(original_image, size_training)
    if flag == 'unet':
        processed_image = preprocessing_agumentation(original_image, size_training)
    if flag == 'unet_clstm':
        processed_image = preprocessing_agumentation(original_image, size_training)
    if flag == 'unet_biclstm':
        processed_image = preprocessing_agumentation(original_image, size_training)
    smoother = niismooth.Smoothutil()
    half_num_slice = n_neighboringslices // 2
    image_input = np.zeros(
        [1, size_training, size_training, n_neighboringslices], dtype=np.float32)
    whole_prediction = []
    for slice_idx in range(processed_image.shape[2]):
        # print('%d / %d ' % (slice_idx, processed_image.shape[2]))
        for j in range(n_neighboringslices):
            cur_idx = slice_idx - half_num_slice + j
            if cur_idx < 0:
                cur_idx = 0
            if cur_idx >= processed_image.shape[2]:
                cur_idx = processed_image.shape[2] - 1

            image_input[0, :, :, j] = processed_image[:, :, cur_idx]
        print('The processed_image shape is ', np.shape(processed_image))
        print('The batch_image shape is ', np.shape(image_input))
        pred = prediction_one_batch(model, image_input, interp, cuda_gpu)
        if optimizer:
            pred = torch.nn.Softmax()(torch.Tensor(pred)).cpu().data.numpy()
            # print('After softmax the pred: ', np.shape(pred))
            # pred = np.transpose(pred, axes=[0, 2, 1])
            resized_predictions = pred[1, :, :]
            resized_convolved = smoother.convolve2d(pred[1, :, :])
            quantified = np.zeros([original_size, original_size], np.uint8)
            quantified[resized_convolved > 0.51] = 1
            quantified[resized_convolved <= 0.51] = 0
            quantified[resized_predictions <= 0.44] = 0
            quantified[resized_predictions >= 0.57] = 1
            pred = quantified
            # pred = np.transpose(quantified, axes=[1, 0])
        else:
            pred = np.argmax(pred, axis=0)
        whole_prediction.append(pred)
    whole_prediction = np.asarray(np.transpose(whole_prediction, axes=[1, 2, 0]), np.uint8)
    gt_image = np.asarray(gt_image, np.uint8)
    print('The shape of whole_prediction is: ', np.shape(whole_prediction))
    print('The shape of GT_image is: ', np.shape(gt_image))
    num_intersection = np.sum(np.logical_and(whole_prediction >= 1, gt_image >= 1))
    num_ground_truth = np.sum(gt_image >= 1)
    num_prediction = np.sum(whole_prediction >= 1)
    dice = (2 * num_intersection + 1e-7) / (num_ground_truth + num_prediction + 1e-7)
    print('Intersection: %d, ground_truth: %d, prediction: %d. Dice: %.4f' % (
        num_intersection, num_ground_truth, num_prediction, dice))
    return dice


def evaluate_measure_one_dir(img_dir, gt_dir, restore_path, size_training, n_neighboringslices, batch_size=1, flag='original'):
    from glob import glob
    img_paths = glob(os.path.join(img_dir, 'volume-*.nii'))
    print(os.path.join(img_dir, 'volume-*.nii'))
    total_dice = 0.0
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        file_id = basename.split('-')[1]
        gt_path = os.path.join(gt_dir, 'segmentation-' + file_id)
        dice = evaluate_measure_one_case(img_path, gt_path, restore_path, size_training, n_neighboringslices, batch_size, flag=flag)
        print(basename, ': ', dice)
        total_dice += dice
    print('Average dice: ', total_dice / len(img_paths))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    model = Res_Deeplab(num_classes=args.num_classes)

    if args.pretrained_model != None:
        args.restore_from = pretrianed_models_dict[args.pretrained_model]

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
    data_list = []

    colorize = VOCColorize()

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        print('image shape: ', np.shape(image))
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output).cpu().data[0].numpy()

        output = output[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        # show_all(gt, output)
        data_list.append([gt.flatten(), output.flatten()])

    filename = os.path.join(args.save_dir, 'result.txt')
    get_iou(data_list, args.num_classes, filename)


if __name__ == '__main__':
    # main()
    evaluate_measure_one_case(
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2/volume-28.nii',
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2/segmentation-28.nii',
        restore_path=None,
        size_training=400, n_neighboringslices=5, batch_size=1, flag='unet_biclstm'
    )


    # if for unet n_neighboringslices is
    # evaluate_measure_one_dir(
    #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1',
    #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1',
    #     '/home/give/github/AdvSemiSeg/snapshots/Liver1947.pth',
    #     size_training=400, n_neighboringslices=5, batch_size=1, flag='unet_clstm'
    # )
# -*- coding=utf-8 -*-
import SimpleITK as itk
import pydicom
import numpy as np
from PIL import Image, ImageDraw
import gc
from skimage.morphology import disk, dilation
import nipy
import os
import scipy
import cv2


typenames = ['CYST', 'FNH', 'HCC', 'HEM', 'METS']
typeids = [0, 1, 2, 3, 4]


def get_voxel_size(file_path):
    load_image_obj = nipy.load_image(file_path)
    header = load_image_obj.header
    x_size = header['srow_x'][0]
    y_size = header['srow_y'][1]
    z_size = header['srow_z'][2]
    return [x_size, y_size, z_size]


def read_nil(file_path):
    return nipy.load_image(file_path).get_data()


# 读取文件序列
def read_dicom_series(dir_name):
    reader = itk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dir_name)
    reader.SetFileNames(dicom_series)
    images = reader.Execute()
    image_array = itk.GetArrayFromImage(images)
    return image_array


# 将DICOM序列转化成MHD文件
def convert_dicomseries2mhd(dicom_series_dir, save_path):
    data = read_dicom_series(dicom_series_dir)
    save_mhd_image(data, save_path)


# 读取单个DICOM文件
def read_dicom_file(file_name):
    header = pydicom.read_file(file_name)
    image = header.pixel_array
    image = header.RescaleSlope * image + header.RescaleIntercept
    return image


# 读取mhd文件
def read_mhd_image(file_path, rejust=False):
    header = itk.ReadImage(file_path)
    image = np.array(itk.GetArrayFromImage(header))
    if rejust:
        image[image < -70] = -70
        image[image > 180] = 180
        image = image + 70
    return np.array(image)


# 保存mhd文件
def save_mhd_image(image, file_name):
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, file_name)


# 根据文件名返回期项名
def return_phasename(file_name):
    phasenames = ['NC', 'ART', 'PV']
    for phasename in phasenames:
        if file_name.find(phasename) != -1:
            return phasename


# 读取DICOM文件中包含的病例ID信息
def read_patientId(dicom_file_path):
    ds = pydicom.read_file(dicom_file_path)
    return ds.PatientID


# 返回病灶类型和ID的字典类型的数据 key是typename value是typeid
def return_type_nameid():
    res = {}
    res['CYST'] = 0
    res['FNH'] = 1
    res['HCC'] = 2
    res['HEM'] = 3
    res['METS'] = 4
    return res


# 返回病灶类型ID和名称的字典类型的数据 key是typeid value是typename
def return_type_idname():
    res = {}
    res[0] = 'CYST'
    res[1] = 'FNH'
    res[2] = 'HCC'
    res[3] = 'HEM'
    res[4] = 'METS'
    return res


# 根据病灶类型的ID返回类型的字符串
def return_typename_byid(typeid):
    idname_dict = return_type_idname()
    return idname_dict[typeid]


# 根据病灶类型的name返回id的字符串
def return_typeid_byname(typename):
    nameid_dict = return_type_nameid()
    return nameid_dict[typename]


# 填充图像
def fill_region(image):
    # image.show()
    from scipy import ndimage
    image = ndimage.binary_fill_holes(image).astype(np.uint8)
    return image


def open_operation(slice_image, kernel_size=7):
    opening = cv2.morphologyEx(slice_image, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    return opening
# 图像膨胀
# def image_expand(image, size):
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
#    image = cv2.dilate(image, kernel)
#    return image


# 将一个矩阵保存为图片
def save_image(image_arr, save_path):
    image = Image.fromarray(np.asarray(image_arr, np.uint8))
    image.save(save_path)


def show_image(image):
    img = Image.fromarray(np.asarray(image, np.uint8))
    img.show()


# 将图像画出来，并且画出标记的病灶
def save_image_with_mask(image_arr, mask_image, save_path):
    image_arr[image_arr < -70] = -70
    image_arr[image_arr > 180] = 180
    image_arr = image_arr + 70
    shape = list(np.shape(image_arr))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = image_arr
    image_arr_rgb[:, :, 1] = image_arr
    image_arr_rgb[:, :, 2] = image_arr
    image = Image.fromarray(np.asarray(image_arr_rgb, np.uint8))
    image_draw = ImageDraw.Draw(image)
    [ys, xs] = np.where(mask_image != 0)
    miny = np.min(ys)
    maxy = np.max(ys)
    minx = np.min(xs)
    maxx = np.max(xs)
    ROI = image_arr_rgb[miny - 1:maxy + 1, minx - 1:maxx + 1, :]
    ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))

    for index, y in enumerate(ys):
        image_draw.point([xs[index], y], fill=(255, 0, 0))
    if save_path is None:
        image.show()
    else:
        image.save(save_path)
        ROI_Image.save(os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0] + '_ROI.jpg'))
        del image, ROI_Image
        gc.collect()


def compress22dim(image):
    '''
        将一个矩阵如果可能，压缩到三维的空间
    '''
    shape = list(np.shape(image))
    if len(shape) == 3:
        return np.squeeze(image)
    return image


def extract_ROI(image, mask_image):
    '''
        提取一幅图像中的ＲＯＩ
    '''
    xs, ys = np.where(mask_image == 1)
    xs_min = np.min(xs)
    xs_max = np.max(xs)
    ys_min = np.min(ys)
    ys_max = np.max(ys)
    return image[xs_min: xs_max + 1, ys_min: ys_max + 1]


def resize_image(image, size):
    image = Image.fromarray(np.asarray(image, np.uint8))
    return image.resize((size, size))


def image_expand(mask_image, r):
    return dilation(mask_image, disk(r))


'''
    将形式如(512, 512)格式的图像转化为(1, 512, 512)形式的图片
'''
def expand23D(mask_image):
    shape = list(np.shape(mask_image))
    if len(shape) == 2:
        mask_image = np.expand_dims(mask_image, axis=0)
        print('after expand23D', np.shape(mask_image))
    return mask_image


'''
    返回一个ｍａｓｋ图像的中心，是对ｘｙｚ坐标计算平均值之后的结果
'''
def find_centroid3D(image, flag):
    [x, y, z] = np.where(image == flag)
    centroid_x = int(np.mean(x))
    centroid_y = int(np.mean(y))
    centroid_z = int(np.mean(z))
    return centroid_x, centroid_y, centroid_z


'''
    将[w, h, d]reshape为[d, w, h]
'''
def convert2depthfirst(image):
    image = np.array(image)
    shape = np.shape(image)
    new_image = np.zeros([shape[2], shape[0], shape[1]])
    for i in range(shape[2]):
        new_image[i, :, :] = image[:, :, i]
    return new_image
    # def test_convert2depthfirst():
    #     zeros = np.zeros([100, 100, 30])
    #     after_zeros = convert2depthfirst(zeros)
    #     print np.shape(after_zeros)
    # test_convert2depthfirst()

'''
    将[d, w, h]reshape为[w, h, d]
'''
def convert2depthlastest(image):
    image = np.array(image)
    shape = np.shape(image)
    new_image = np.zeros([shape[1], shape[2], shape[0]])
    for i in range(shape[0]):
        new_image[:, :, i] = image[i, :, :]
    return new_image


def read_image_file(file_path):
    if file_path.endswith('.nii'):
        return read_nil(file_path)
    if file_path.endswith('.mhd'):
        return read_mhd_image(file_path)
    print('the format of image is not support in this version')
    return None


def processing(image, size_training):
    image = np.array(image)
    # numpy_clip
    bottom = -300.
    top = 500.
    image = np.clip(image, bottom, top)

    # to float
    minval = -350
    interv = 500 - (-350)
    image -= minval
    # scale down to 0 - 2
    image /= (interv / 2)

    # zoom
    desired_size = [size_training, size_training]
    desired_size = np.asarray(desired_size, dtype=np.int)
    zooms = desired_size / np.array(image[:, :, 0].shape, dtype=np.float)
    print(zooms)
    after_zoom = np.zeros([size_training, size_training, np.shape(image)[2]])
    for i in range(np.shape(after_zoom)[2]):
        after_zoom[:, :, i] = scipy.ndimage.zoom(image[:, :, i], zooms, order=1)  # order = 1 => biliniear interpolation

    return after_zoom

if __name__ == '__main__':
    # for phasename in ['NC', 'ART', 'PV']:
    #     convert_dicomseries2mhd(
    #         '/home/give/github/Cascaded-FCN-Tensorflow/Cascaded-FCN/tensorflow-unet/z_testdata/304176-2802027/' + phasename,
    #         '/home/give/github/Cascaded-FCN-Tensorflow/Cascaded-FCN/tensorflow-unet/z_testdata/304176-2802027/MHD/' + phasename + '.mhd'
    #     )

    names = os.listdir('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2')
    for name in names:
        path = os.path.join('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2', name)
        image = read_nil(path)
        print(np.shape(image))
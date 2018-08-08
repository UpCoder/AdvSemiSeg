import preprocessing
import tensorflow as tf
import numpy as np
from medicalImage import save_mhd_image
train_data_dir = '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1'
validation_interval = 100
max_steps = 1000000000
batch_size = 4
n_neighboringslices = 5
input_size = 320
output_size = 320
oversample = False
slice_type = 'axial'
load_path = None
reset_counter = False
summaries_dir = None
snapshot_path = None
label_of_interest = 1
label_required = 0
magic_number = 7.5
max_slice_tries_val = 100
max_slice_tries_train = 100
fuse_labels = True
apply_crop = False
training_pipeline = preprocessing.training(
                train_data_dir,
                slice_type=slice_type,
                n_neighboringslices=n_neighboringslices,
                image_size=input_size,
                oversample=oversample,
                label_of_interest=label_of_interest,
                label_required=label_required,
                max_tries=max_slice_tries_train,
                fuse_labels=fuse_labels,
                apply_crop=apply_crop,
                data_augmentation=True
            )
source = training_pipeline.run_on(1, num_datapoints=(validation_interval))
for i in range(10):
    input_image, _ = source.next()
    input_image = np.asarray(input_image)
    print(i, np.shape(input_image), np.max(input_image), np.min(input_image))
    save_mhd_image(input_image[:, 0, :, :],
                   './test/' + str(i) + '.nii')
    save_mhd_image(input_image[:, 1, :, :],
                   './test/' + str(i) + '_gt.nii')

# ('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1', 'axial', 1, 320, False, 1, 0, 100, 1.0, True, False, False)
# ('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2', 'axial', 1, 400, False, 1, 0, 2, 1.0, True, False, True)
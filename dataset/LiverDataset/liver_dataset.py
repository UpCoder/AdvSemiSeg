import torch
from torch.utils import data
import numpy as np
import time
from dataset.LiverDataset import preprocessing, medicalImage


class LiverDatasetBase(data.Dataset):
    def __init__(self, data_dir, slice_type, n_neighboringslices, input_size, oversample, label_of_interest,
                 label_required, max_slice_tries, fuse_labels, apply_crop, validation_interval, is_training,
                 data_augmentation):
        self.validation_interval = validation_interval
        self.dataset_len = validation_interval
        self.input_size = input_size
        self.slice_type = slice_type
        self.n_neighboringslices = n_neighboringslices
        self.label_idx = self.n_neighboringslices // 2
        if is_training:
            self.pipeline = preprocessing.training(
                data_dir,
                slice_type=slice_type,
                n_neighboringslices=n_neighboringslices,
                image_size=input_size,
                oversample=oversample,
                label_of_interest=label_of_interest,
                label_required=label_required,
                max_tries=max_slice_tries,
                fuse_labels=fuse_labels,
                apply_crop=apply_crop,
                data_augmentation=data_augmentation
            )
        else:
            self.pipeline = preprocessing.validation(
                data_dir,
                slice_type=slice_type,
                n_neighboringslices=n_neighboringslices,
                image_size=input_size,
                label_of_interest=label_of_interest,
                label_required=label_required,
                max_tries=max_slice_tries,
                fuse_labels=fuse_labels,
                apply_crop=apply_crop
            )

        self.source = self.pipeline.run_on(1, num_datapoints=(self.validation_interval))

    def restart(self):
        self.source = self.pipeline.run_on(1, num_datapoints=(self.validation_interval))

    def __getitem__(self, index):
        inputs, _ = self.source.next()
        image_input = np.zeros([self.input_size, self.input_size, self.n_neighboringslices],
                               dtype=np.float32)
        for j in range(self.n_neighboringslices):
            image_input[:, :, j] = inputs[j][0]
        mask = inputs[self.label_idx][1]
        return image_input, mask

    def __len__(self):
        return self.validation_interval


class LiverDataset:
    def __init__(self, data_dir, slice_type, n_neighboringslices, input_size, oversample, label_of_interest,
                 label_required, max_slice_tries, fuse_labels, apply_crop, interval, is_training, batch_size,
                 data_augmentation=False):
        self.batch_size = batch_size
        self.datasetbase = LiverDatasetBase(
            data_dir=data_dir,
            slice_type=slice_type,
            n_neighboringslices=n_neighboringslices,
            input_size=input_size,
            oversample=oversample,
            label_of_interest=label_of_interest,
            label_required=label_required,
            max_slice_tries=max_slice_tries,
            fuse_labels=fuse_labels,
            apply_crop=apply_crop,
            validation_interval=interval,
            is_training=is_training,
            data_augmentation=data_augmentation
        )
        self.loader = torch.utils.data.DataLoader(self.datasetbase, batch_size=batch_size,
                                                  shuffle=True, worker_init_fn=1,
                                                  pin_memory=True)
        self.loader_iter = enumerate(self.loader)

    def get_next_batch(self):
        while True:
            try:
                _, (batch_image, batch_mask) = self.loader_iter.next()
            except Exception, e:
                self.datasetbase.restart()
                self.loader_iter = enumerate(self.loader)
                _, (batch_image, batch_mask) = self.loader_iter.next()
            if len(batch_image) != self.batch_size:
                continue
            else:
                break
        return batch_image, batch_mask



if __name__ == '__main__':
    user_name = 'ld'
    validation_interval = 10
    max_steps = 1000000000
    batch_size = 3
    n_neighboringslices = 5
    input_size = 400
    output_size = 400
    slice_type = 'axial'
    oversample = False
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
        interval=validation_interval, is_training=True, batch_size=batch_size
    )
    val_dataset = LiverDataset(
        data_dir=test_data_dir, slice_type=slice_type, n_neighboringslices=n_neighboringslices, input_size=input_size,
        oversample=oversample, label_of_interest=label_of_interest, label_required=label_required,
        max_slice_tries=max_slice_tries_val, fuse_labels=fuse_labels, apply_crop=apply_crop,
        interval=validation_interval, is_training=False, batch_size=batch_size
    )
    for i in range(1000):
        batch_image, batch_mask = train_dataset.get_next_batch()
        print('Image: ', np.shape(batch_image), ' Mask: ', np.shape(batch_mask))
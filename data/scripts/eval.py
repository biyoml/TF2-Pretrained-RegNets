import os
import argparse
import glob
import scipy.io as sio
import tensorflow as tf


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, dct_method='INTEGER_ACCURATE')
    return tf.image.convert_image_dtype(image, tf.float32)


def center_crop(image, resize_size=256, crop_size=224):
    imsize = tf.cast(tf.shape(image)[:2], tf.float32)
    smaller_edge = tf.reduce_min(imsize)
    imsize = tf.round(imsize * resize_size / smaller_edge)
    image = tf.image.resize(image, tf.cast(imsize, tf.int32), antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
    return image


def normalize(image, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225]):
    return (image - mean) / stddev


def parse_meta_mat():
    meta = sio.loadmat('data/meta.mat', squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids = list(zip(*meta))[:2]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    return wnids, idx_to_wnid


def load_dataset(data_dir, batch_size):
    images = glob.glob(os.path.join(data_dir, '*.JPEG'))
    images.sort()

    wnids, idx_to_wnid = parse_meta_mat()
    wnids = sorted(list(wnids))
    with open('data/ILSVRC2012_validation_ground_truth.txt') as f:
        val_idcs = [int(line.strip()) for line in f.readlines()]
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    labels = [wnids.index(wnid) for wnid in val_wnids]

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (load_image(x), y))
    dataset = dataset.map(lambda x, y: (center_crop(x), y))
    dataset = dataset.map(lambda x, y: (normalize(x), y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h5', type=str, required=True,
                        help="converted model")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="a folder containing ImageNet validation images")
    parser.add_argument('--batch_size', type=int, required=True,
                        help="batch size")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir, args.batch_size)
    model = tf.keras.models.load_model(args.h5)
    model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.evaluate(dataset)


if __name__ == '__main__':
    main()

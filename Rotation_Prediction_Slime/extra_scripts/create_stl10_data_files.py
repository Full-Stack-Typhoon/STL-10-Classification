import argparse
import errno
import os
import sys
import tarfile

import numpy as np
from urllib.request import urlretrieve
try:
    from imageio import imsave
except:
    from scipy.misc import imsave


DEFAULT_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'


def get_args():
    def restricted_ratio(x):
        x = float(x)
        if (x < 0.0) or (x > 1.0):
            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
        return x
    parser = argparse.ArgumentParser(
        description="Downloads and extracts the STL-10 dataset."
    )
    parser.add_argument('--dest_dir', type=str, default="data",
                        help="Path to the destination directory")
    parser.add_argument('--data_url', type=str, default=DEFAULT_URL,
                        help="Default STL-10 URL")
    parser.add_argument('--unsupervised_split', type=restricted_ratio, default=0.9,
                        help="Ratio for Unsupervised Training")
    return parser.parse_args()


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return np.asarray(labels-1, dtype=np.long)


def read_images_bin(path_to_bin_file):
    with open(path_to_bin_file, 'rb') as fobj:
        images = np.fromfile(fobj, dtype=np.uint8)
        images = images.reshape(-1, 3, 96, 96)
        images = images.transpose(0, 3, 2, 1)
        return images

def save_images(directory, images):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
    X = []
    for i, image in enumerate(images):
        filename = os.path.join(directory, f"{i}.png")
        print(filename)
        imsave(filename, image, format="png")
        X.append(filename)
    return np.asarray(X)

def download_and_extract(args):
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    filename = os.path.basename(args.data_url)
    filepath = os.path.join(args.dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            percentage_completed = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write('\rDownloaded %s %.2f%%' % (filename,percentage_completed))
            sys.stdout.flush()
        filepath, _ = urlretrieve(args.data_url, filepath, reporthook=_progress)
        print('Completed Download', filename)
        tarfile.open(filepath, 'r:gz').extractall(args.dest_dir)
    return filepath[:-7]

def main():
    args = get_args()
    stl10_dir = download_and_extract(args)

    # Train
    train_bin_images = read_images_bin(os.path.join(stl10_dir, 'train_X.bin'))
    train_bin_labels = read_labels(os.path.join(stl10_dir, 'train_y.bin'))
    fold_indices = open(os.path.join(stl10_dir, 'fold_indices.txt'), 'r').readlines()
    fold_indices = [ list(map(lambda x: int(x),line.strip().split(' '))) for line in fold_indices]
    print('Saving train images')
    train_prefix = os.path.join(args.dest_dir, 'train')
    train_all = save_images(train_prefix, train_bin_images)
    for idx, arr in enumerate(fold_indices):
        np.save(os.path.join(args.dest_dir, f'train_images_{idx}.npy'), train_all[arr[:]])
        print(train_bin_labels.shape)
        np.save(os.path.join(args.dest_dir, f'train_labels_{idx}.npy'), train_bin_labels[arr[:]])

    # Test
    print('Saving test images')
    test_images = read_images_bin(os.path.join(stl10_dir, 'test_X.bin'))
    test_labels = read_labels(os.path.join(stl10_dir, 'test_y.bin'))
    test_X = save_images(os.path.join(args.dest_dir,'test'), test_images)
    np.save(os.path.join(args.dest_dir, 'test_images.npy'), test_X)
    np.save(os.path.join(args.dest_dir, 'test_labels.npy'), test_labels)

    # Selfsupervised
    print('Saving self supervised images')
    unsupervised = read_images_bin(os.path.join(stl10_dir, 'unlabeled_X.bin'))
    unsupervised_train = int(args.unsupervised_split*len(unsupervised))
    unlabeled_X_train = save_images(os.path.join(args.dest_dir,'unsupervised_train'), unsupervised[:unsupervised_train])
    unlabeled_X_test = save_images(os.path.join(args.dest_dir,'unsupervised_test'), unsupervised[unsupervised_train:])

    np.save(os.path.join(args.dest_dir, 'unsupervised_train_images.npy'), unlabeled_X_train)
    np.save(os.path.join(args.dest_dir, 'unsupervised_test_images.npy'), unlabeled_X_test)

if __name__ == '__main__':
    main()

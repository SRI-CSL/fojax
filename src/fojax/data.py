import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
import jax.numpy as jnp
import os
import pickle
import urllib.request
import tarfile
import numpy as np

def download_cifar10(destination="./data"):
    """
    Download and extract the CIFAR-10 dataset (if it doesn't exist already).
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_tar = os.path.join(destination, "cifar-10-python.tar.gz")
    cifar_dir = os.path.join(destination, "cifar-10-batches-py")
    
    if not os.path.exists(cifar_dir):
        os.makedirs(destination, exist_ok=True)
        print("Downloading CIFAR-10 (approx 170MB)...")
        urllib.request.urlretrieve(url, cifar_tar)
        print("Extracting...")
        with tarfile.open(cifar_tar, "r:gz") as tar:
            tar.extractall(path=destination)
        print("Done.")
    else:
        print("CIFAR-10 already downloaded.")

def load_cifar10_numpy(data_dir="./data/cifar-10-batches-py"):
    """
    Returns:
      x_train: NumPy array of shape (50000, 32, 32, 3), dtype=np.uint8
      y_train: NumPy array of shape (50000,)
      x_test:  NumPy array of shape (10000, 32, 32, 3)
      y_test:  NumPy array of shape (10000,)
    """
    # CIFAR-10 is split into 5 training batches + 1 test batch
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        x_train.append(batch["data"])
        y_train.extend(batch["labels"])
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    # Reshape flat pixel data into (N, 32, 32, 3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0,2,3,1)

    # Load test batch
    test_path = os.path.join(data_dir, "test_batch")
    with open(test_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    x_test = batch["data"].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
    y_test = np.array(batch["labels"])

    return x_train, y_train, x_test, y_test

def random_flip_left_right(img, rng):
    if rng.rand() < 0.5:
        return np.fliplr(img)
    return img

def random_crop_reflect(img, pad=4, rng=None):
    """
    Pad the image by `pad` pixels on each side, then randomly crop back to 32x32.
    Reflect-pad is equivalent to 'reflect' mode in TensorFlow.
    """
    if rng is None:
        rng = np.random
    # pad: shape -> (32 + 2*pad, 32 + 2*pad, 3)
    img_padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    # random crop indices
    start_x = rng.randint(0, pad*2)  # e.g. 0..8 if pad=4
    start_y = rng.randint(0, pad*2)
    return img_padded[start_x:start_x+32, start_y:start_y+32, :]

def normalize(img, mean, std):
    return (img - mean) / std

class CIFAR10DataGenerator:
    """
    A custom generator that yields dictionaries:
       {"image": ..., "label": ...}
    """
    def __init__(self, x, y, batch_size, shuffle=True, augment=False, seed=0):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.num_samples = len(x)
        self.indexes = np.arange(self.num_samples)

        # Normalization constants
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        self.std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

    def as_numpy_iterator(self):
        """
        Infinite generator that mimics tf.data.Dataset.repeat().batch().shuffle() usage.
        Each yield is a dict: {"image": numpy array, "label": numpy array}
        """
        while True:
            if self.shuffle:
                self.rng.shuffle(self.indexes)

            # Slice batches
            for start_idx in range(0, self.num_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                if end_idx > self.num_samples:
                    break

                batch_indices = self.indexes[start_idx:end_idx]
                images = self.x[batch_indices].astype(np.float32) / 255.0  # scale to [0,1]
                labels = self.y[batch_indices]

                # Apply augmentations if training
                if self.augment:
                    for i in range(len(images)):
                        images[i] = random_crop_reflect(images[i], pad=4, rng=self.rng)
                        images[i] = random_flip_left_right(images[i], self.rng)

                # Normalize each image: (img - mean) / std
                # Do it per-pixel-channel
                for i in range(len(images)):
                    images[i] = normalize(images[i], self.mean, self.std)

                yield {"image": images, "label": labels}


def get_cifar10_dataloaders_no_tf(batch_size=256, seed=0, buffer=1024):
    """
    Equivalent to the TF-based function but using only raw Python + NumPy.
    Returns:
      train_ds, test_ds, steps_per_epoch
    where train_ds/test_ds have a .as_numpy_iterator() method that yields dicts.
    """
    # Download (if needed) and load the CIFAR-10 data
    download_cifar10("./data")
    x_train, y_train, x_test, y_test = load_cifar10_numpy("./data/cifar-10-batches-py")

    # Steps per epoch (like in TF)
    steps_per_epoch_train = len(x_train) // batch_size
    steps_per_epoch_test = len(x_test) // batch_size

    # Create data generators
    train_ds = CIFAR10DataGenerator(
        x_train, y_train, batch_size=batch_size,
        shuffle=True, augment=True, seed=seed
    )
    test_ds = CIFAR10DataGenerator(
        x_test,  y_test,  batch_size=batch_size,
        shuffle=False, augment=False, seed=seed
    )

    return train_ds, test_ds, steps_per_epoch_train, steps_per_epoch_test

def get_mnist_dataloaders(batch_size=128, seed=0, buffer=1024):
    """
    Load and preprocess the MNIST dataset into TensorFlow dataloaders.
    
    Parameters:
    - batch_size (int): The batch size for training and testing.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - train_ds (tf.data.Dataset): Preprocessed training dataset.
    - test_ds (tf.data.Dataset): Preprocessed test dataset.
    """
    # Set the random seed for reproducibility
    tf.random.set_seed(seed)
    
    # Load dataset info without creating the dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    ds_info = ds_builder.info
    
    # Get number of training examples
    num_train_examples = ds_info.splits['train'].num_examples
    steps_per_epoch = num_train_examples // batch_size
    print(f"Number of training examples: {num_train_examples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Load train and test datasets
    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')
    
    # Preprocess datasets
    def preprocess(sample):
        return {
            'image': tf.cast(sample['image'], tf.float32) / 255.0,
            'label': sample['label'],
        }
    
    train_ds = (
        train_ds
        .map(preprocess)
        .shuffle(buffer, seed=seed)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    
    test_ds = (
        test_ds
        .map(preprocess)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    
    return train_ds, test_ds, steps_per_epoch

def get_cifar10_dataloaders(batch_size=256, seed=0, buffer=1024):
    """
    Load and preprocess the CIFAR-10 dataset into TensorFlow dataloaders.
    
    Parameters:
    - batch_size (int): The batch size for training and testing.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - train_ds (tf.data.Dataset): Preprocessed training dataset.
    - test_ds (tf.data.Dataset): Preprocessed test dataset.
    """
    # Set the random seed for reproducibility
    tf.random.set_seed(seed)
    
    # Load dataset info without creating the dataset
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    ds_info = ds_builder.info
    
    # Get number of training examples
    num_train_examples = ds_info.splits['train'].num_examples
    steps_per_epoch = num_train_examples // batch_size
    print(f"Number of training examples: {num_train_examples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # CIFAR-10 normalization values
    mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
    std = tf.constant([0.2470, 0.2435, 0.2616], dtype=tf.float32)
    
    def preprocess_train(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0
        image = tf.pad(image, paddings=[[4, 4], [4, 4], [0, 0]], mode='REFLECT')
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        image = (image - mean) / std
        return {'image': image, 'label': sample['label']}
    
    def preprocess_test(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0
        image = (image - mean) / std
        return {'image': image, 'label': sample['label']}
    
    # Load train and test datasets
    train_ds = tfds.load('cifar10', split='train')
    test_ds = tfds.load('cifar10', split='test')
    
    train_ds = (
        train_ds
        .map(preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(buffer, seed=seed)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    
    test_ds = (
        test_ds
        .map(preprocess_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    
    return train_ds, test_ds, steps_per_epoch

# ---------------------------------------------------------------------
# MNIST (pure NumPy) utilities ────────────────────────────────────────
# ---------------------------------------------------------------------
import urllib.request, pathlib, gzip, shutil

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def download_mnist_npz(destination="./data/mnist"):
    """
    Download `mnist.npz` once.  Destination defaults to ./data/mnist.
    """
    dest = pathlib.Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    npz_path = dest / "mnist.npz"

    if not npz_path.exists():
        print("Downloading MNIST (~11 MB)…")
        urllib.request.urlretrieve(MNIST_URL, npz_path)
        print(f"Saved to {npz_path}")
    else:
        print("MNIST already downloaded.")

    return npz_path


def load_mnist_numpy(destination="/tmp/data/mnist"):
    """
    Returns:
      x_train: (60000, 28, 28, 1) uint8
      y_train: (60000,)          uint8
      x_test:  (10000, 28, 28, 1) uint8
      y_test:  (10000,)          uint8
    """
    npz_path = download_mnist_npz(destination)
    with np.load(npz_path) as data:
        x_train = data["x_train"][..., None]  # add channel dim
        y_train = data["y_train"]
        x_test  = data["x_test"][..., None]
        y_test  = data["y_test"]
    return x_train, y_train, x_test, y_test


# ------------------------------------------------------------
# Augment / preprocess helpers (NumPy)
# ------------------------------------------------------------
def random_flip_left_right(img, rng):
    return np.fliplr(img) if rng.rand() < 0.5 else img

def random_pad_crop(img, pad=2, rng=None):
    if rng is None:
        rng = np.random
    img_p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    x0, y0 = rng.randint(0, pad * 2 + 1, size=2)   # inclusive
    return img_p[x0:x0+28, y0:y0+28, :]

def normalize(img, mean, std):
    return (img - mean) / std


class MNISTDataGenerator:
    """
    Yields {'image': imgs, 'label': labels} exactly like the CIFAR version.
    """
    def __init__(self, x, y, batch_size, shuffle=True, augment=False, seed=0):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.num_samples = len(x)
        self.indexes = np.arange(self.num_samples)

        self.mean = np.float32(0.1307)
        self.std  = np.float32(0.3081)

    def __iter__(self):
        return self.as_numpy_iterator()

    def as_numpy_iterator(self):
        while True:
            if self.shuffle:
                self.rng.shuffle(self.indexes)

            for start in range(0, self.num_samples, self.batch_size):
                end = start + self.batch_size
                if end > self.num_samples:               # drop last incomplete
                    break

                idx = self.indexes[start:end]
                imgs = self.x[idx].astype(np.float32) / 255.0
                labels = self.y[idx]

                if self.augment:
                    for i in range(len(imgs)):
                        imgs[i] = random_pad_crop(imgs[i], pad=2, rng=self.rng)
                        imgs[i] = random_flip_left_right(imgs[i], self.rng)

                for i in range(len(imgs)):
                    imgs[i] = normalize(imgs[i], self.mean, self.std)

                yield {"image": imgs, "label": labels}



def get_mnist_dataloaders_no_tf(batch_size=128, seed=0, augment =False):
    """
    Pure-NumPy MNIST loader matching the signature of get_cifar10_dataloaders_no_tf.
    Returns (train_ds, test_ds, steps_per_epoch_train, steps_per_epoch_test).
    """
    x_train, y_train, x_test, y_test = load_mnist_numpy("./data/mnist")

    steps_train = len(x_train) // batch_size
    steps_test  = len(x_test)  // batch_size

    train_ds = MNISTDataGenerator(
        x_train, y_train,
        batch_size=batch_size,
        shuffle=True, augment=augment, seed=seed,
    )
    test_ds = MNISTDataGenerator(
        x_test, y_test,
        batch_size=batch_size,
        shuffle=False, augment=False, seed=seed,
    )

    return train_ds, test_ds, steps_train, steps_test


from flax import nnx
import jax
from functools import partial
import jax.numpy as jnp

class LogReg(nnx.Module):
    """Simple logistic regression layer."""
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass; returns logits (no softmax)."""
        return self.linear(x)

class MLP(nnx.Module):
    """An MLP similar to the PyTorch Net."""
    def __init__(self, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(1,   100, rngs=rngs)
        self.fc2 = nnx.Linear(100, 100, rngs=rngs)
        self.fc3 = nnx.Linear(100, 100, rngs=rngs)
        self.fc4 = nnx.Linear(100, 100, rngs=rngs)
        self.fc5 = nnx.Linear(100, 100, rngs=rngs)
        self.fc6 = nnx.Linear(100,   1, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        x = jax.nn.relu(self.fc3(x))
        x = jax.nn.relu(self.fc4(x))
        x = jax.nn.relu(self.fc5(x))
        x = self.fc6(x)  # Final layer is linear
        return x
        

class CNN_MNIST(nnx.Module):
  """A simple CNN model."""
  def __init__(self, activation: nnx.relu, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)
    self.activation = activation

  def __call__(self, x):
    x = self.avg_pool(self.activation(self.conv1(x)))
    x = self.avg_pool(self.activation(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = self.activation(self.linear1(x))
    x = self.linear2(x)
    return x

class CNN_CIFAR(nnx.Module):
  def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
    # --- Block 1 (conv_layer_b1) ---
    self.conv1_1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn1_1   = nnx.BatchNorm(32, rngs=rngs)
    self.conv1_2 = nnx.Conv(32, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn1_2   = nnx.BatchNorm(32, rngs=rngs)
    # MaxPool with window (2,2) and stride (2,2)
    self.maxpool1 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Block 2 (first part of conv_layer_b2) ---
    self.conv2_1 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn2_1   = nnx.BatchNorm(64, rngs=rngs)
    self.conv2_2 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn2_2   = nnx.BatchNorm(64, rngs=rngs)
    self.maxpool2 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Block 3 (second part of conv_layer_b2) ---
    self.conv3_1 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn3_1   = nnx.BatchNorm(128, rngs=rngs)
    self.conv3_2 = nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    self.bn3_2   = nnx.BatchNorm(128, rngs=rngs)
    self.maxpool3 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Fully Connected Layers ---
    # Note: Two separate dropout layers are used, mirroring the original usage.
    # self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)
    self.linear1  = nnx.Linear(128 * 4 * 4, 1024, rngs=rngs)
    # self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)
    self.linear2  = nnx.Linear(1024, num_classes, rngs=rngs)

  def __call__(self, x: jax.Array):
    # --- Block 1 ---
    x = self.conv1_1(x)
    x = self.bn1_1(x)
    x = nnx.relu(x)
    x = self.conv1_2(x)
    x = self.bn1_2(x)
    x = nnx.relu(x)
    x = self.maxpool1(x)

    # --- Block 2 ---
    x = self.conv2_1(x)
    x = self.bn2_1(x)
    x = nnx.relu(x)
    x = self.conv2_2(x)
    x = self.bn2_2(x)
    x = nnx.relu(x)
    x = self.maxpool2(x)

    # --- Block 3 ---
    x = self.conv3_1(x)
    x = self.bn3_1(x)
    x = nnx.relu(x)
    x = self.conv3_2(x)
    x = self.bn3_2(x)
    x = nnx.relu(x)
    x = self.maxpool3(x)

    # Flatten the features
    x = x.reshape(x.shape[0], -1)
    
    # --- Fully Connected Layers ---
    # x = self.dropout1(x)
    x = self.linear1(x)
    x = nnx.relu(x)
    # x = self.dropout2(x)
    x = self.linear2(x)
    return x

# ResNet copying: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
def conv3x3(in_channels, out_channels, stride=1, *, rngs):
    """3x3 convolution with padding"""
    return nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), strides=(stride, stride), padding="SAME", rngs=rngs)

def conv1x1(in_channels, out_channels, stride=1, *, rngs):
    """1x1 convolution for downsampling"""
    return nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), strides=(stride, stride), padding="SAME", rngs=rngs)

class BasicBlock(nnx.Module):
    expansion = 1  # No expansion like in ResNet-18

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, *, rngs):
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = conv3x3(out_channels, out_channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return nnx.relu(out)

class ResNet18(nnx.Module):
    def __init__(self, num_classes=10, *, rngs):
        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=(2, 2), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.relu = nnx.relu
        self.maxpool = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        self.layer1 = self._make_layer(64, 64, 2, stride=1, rngs=rngs)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, rngs=rngs)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, rngs=rngs)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, rngs=rngs)

        self.avgpool = partial(jax.numpy.mean, axis=(1, 2))  # Global average pooling
        self.fc = nnx.Linear(512, num_classes, rngs=rngs)

    def _make_layer(self, in_channels, out_channels, blocks, stride, *, rngs):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nnx.Sequential(
                conv1x1(in_channels, out_channels, stride=stride, rngs=rngs),
                nnx.BatchNorm(out_channels, rngs=rngs),
            )

        layers = [BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample, rngs=rngs)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, rngs=rngs))

        return nnx.Sequential(*layers)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Global average pooling
        x = self.fc(x)
        return x
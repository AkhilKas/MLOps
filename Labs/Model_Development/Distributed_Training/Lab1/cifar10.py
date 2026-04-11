"""
CIFAR-10 dataset and model for distributed training
"""

import tensorflow as tf
import numpy as np


def cifar10_dataset(batch_size, validation_split=0.2):
    """
    Load CIFAR-10 dataset and create train/val splits
    
    Args:
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        
    Returns:
        train_dataset, val_dataset
    """
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Flatten labels
    y_train = y_train.flatten().astype(np.int64)
    y_test = y_test.flatten().astype(np.int64)
    
    # Create train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(50000).repeat().batch(batch_size)
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset


def build_and_compile_cnn_model(learning_rate=0.001):
    """
    Build and compile CNN model for CIFAR-10
    
    Args:
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input: 32x32x3 (CIFAR-10 images)
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        
        # Conv Block 1
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.2),
        
        # Conv Block 2
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Conv Block 3
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.4),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)  # 10 classes
    ])
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model
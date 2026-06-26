import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_DIR = "dataset/asl_alphabet_train/asl_alphabet_train"
OUT_DIR = "dataset/processed"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Dataset direto do diretório (SEM carregar tudo na RAM)
train_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DIR,
    
    validation_split=0.3,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DIR,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# split val/test (50/50 do validation)
val_batches = tf.data.experimental.cardinality(val_test_ds)
val_ds = val_test_ds.take(val_batches // 2)
test_ds = val_test_ds.skip(val_batches // 2)

print("Dataset pronto:")
print(train_ds)
print(val_ds)
print(test_ds)

# Augmentation (aplicado em pipeline, não em RAM)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomFlip("horizontal"),
])

# Otimização de performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("✓ Pipeline otimizado criado com sucesso")
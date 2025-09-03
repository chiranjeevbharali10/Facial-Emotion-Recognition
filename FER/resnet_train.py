
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import sklearn.utils.class_weight
import gc
import signal
import tensorflow.keras.backend as K

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Make Ctrl+C work properly
signal.signal(signal.SIGINT, signal.SIG_DFL)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))
# --- Data Generators ---
# --- Class Weights ---
# Class distribution: sad=4599, disgust=2242, fear=4452, anger=4815, happy=3419, neutral=4474, surprise=4290
class_counts = [4599, 2242, 4452, 4815, 3419, 4474, 4290]
class_labels = np.arange(7)  # changed from list(range(7)) to np.arange(7)
class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=class_labels, y=np.concatenate([
    np.full(c, i) for i, c in enumerate(class_counts)
]))
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print('Class weights:', class_weight_dict)
# --- Enhanced Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,  # increased
    width_shift_range=0.3,  # increased
    height_shift_range=0.3,  # increased
    shear_range=0.3,  # increased
    zoom_range=0.3,  # increased
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # new
    channel_shift_range=30.0,  # new
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# --- Paths ---
train_dir = 'archive/train'
test_dir = 'archive/test'

# --- Flow from directory ---
# NOTE: If you still get OOM errors, reduce batch_size below (e.g., to 16 or 8)
train_dataset = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
    batch_size=8
)

valid_dataset = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    class_mode='categorical',
    subset='validation',
    batch_size=8
)

test_dataset = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16
)

# --- Model Configuration ---
MODEL_TYPE = 'resnet50'  # Change to 'vgg16' to use VGG16 instead
print(f"Using model: {MODEL_TYPE.upper()}")

# --- Load Base Model ---
if MODEL_TYPE.lower() == 'resnet50':
    base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    print("✅ Loaded ResNet50 with ImageNet weights")
elif MODEL_TYPE.lower() == 'vgg16':
    base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    print("✅ Loaded VGG16 with ImageNet weights")
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

# Freeze base layers initially
for layer in base_model.layers:
    layer.trainable = False

# --- Build Model ---
if MODEL_TYPE.lower() == 'resnet50':
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(128, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])
elif MODEL_TYPE.lower() == 'vgg16':
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, kernel_initializer='he_uniform'),  # VGG16 typically uses larger dense layers
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(128, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

model.summary()

# --- Metrics ---
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]
# Add per-class recall metrics
for i in range(7):
    METRICS.append(tf.keras.metrics.Recall(class_id=i, name=f'recall_class_{i}'))

# --- Focal Loss (optional) ---
def focal_loss(gamma=2., alpha=0.25, label_smoothing=0.1):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# --- Compile ---
# LOSS_FN = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)  # label smoothing added
LOSS_FN = focal_loss(gamma=2., alpha=0.25, label_smoothing=0.1)  # focal loss with label smoothing
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=LOSS_FN,
    metrics=METRICS
)

# --- Callbacks ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, class_names):
        super().__init__()
        self.valid_data = valid_data
        self.class_names = class_names
        self.plot_every_n_epochs = 5  # Only plot every 5 epochs to save memory
        
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1} ended. Starting confusion matrix logging...")
        
        # Only create confusion matrix every N epochs to save memory
        if (epoch + 1) % self.plot_every_n_epochs == 0:
            try:
                # Predict in batches, free memory
                y_pred = []
                y_true = []
                
                print("Collecting predictions...")
                for batch_x, batch_y in self.valid_data:
                    batch_pred = self.model.predict_on_batch(batch_x)
                    y_pred.append(batch_pred)
                    y_true.append(batch_y)
                
                y_pred = np.concatenate(y_pred)
                y_true = np.concatenate(y_true)
                
                print("Predictions complete. Now calculating confusion matrix...")
                cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
                
                # Save confusion matrix with proper memory management
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'{MODEL_TYPE.upper()} Confusion Matrix - Epoch {epoch+1}')
                fig.tight_layout()
                plt.savefig(f'{MODEL_TYPE.lower()}_confusion_matrix_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)  # 💡 Important: Close the figure
                
                print("Confusion matrix saved.")
                
                # Save predictions and true labels (optional - comment out if causing issues)
                # np.save(f"y_pred_epoch_{epoch+1}.npy", y_pred)
                # np.save(f"y_true_epoch_{epoch+1}.npy", y_true)
                # print("Numpy arrays saved.")
                
                # Light memory cleanup
                del y_pred, y_true, cm
                gc.collect()
                print("Memory cleaned.")
                
            except Exception as e:
                print(f"Error in confusion matrix callback: {e}")
                # Continue training even if confusion matrix fails
        else:
            print(f"Skipping confusion matrix for epoch {epoch+1} (plotting every {self.plot_every_n_epochs} epochs)")

class_names = list(train_dataset.class_indices.keys())
cm_callback = ConfusionMatrixCallback(valid_dataset, class_names)

# Model-specific checkpoint filename
checkpoint_filename = f'{MODEL_TYPE.lower()}_affectnet.h5'
mcp = ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_accuracy', mode='max')
es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, verbose=1)

# --- Train ---
print("Starting initial training phase...")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=100,
    callbacks=[mcp, es, rlr, cm_callback],
    verbose=1,
    class_weight=class_weight_dict  # added
)

# Clear backend session after initial training
K.clear_session()
print("Initial training completed. Clearing memory...")

# --- Fine-tune: Unfreeze base model and retrain ---
for layer in base_model.layers:
    layer.trainable = True

# Re-compile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=METRICS
)

# Fine-tuning training
print("Starting fine-tuning phase...")
history_finetune = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=30,
    callbacks=[mcp, es, rlr],
    verbose=1,
    class_weight=class_weight_dict  # added
)

# Clear backend session after fine-tuning
K.clear_session()
print("Fine-tuning completed. Clearing memory...")

# --- Evaluate on test dataset ---
test_loss, acc, prec, rec, auc = model.evaluate(test_dataset)
print("\n✅ Test Results:")
print(f"Loss     : {test_loss:.4f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"AUC      : {auc:.4f}")

# After training, plot confusion matrix for test set
print("Evaluating on test set...")
y_true = []
y_pred = []

try:
    for x, y in test_dataset:
        y_true.extend(np.argmax(y, axis=1))
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        if len(y_true) >= test_dataset.samples:
            break
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{MODEL_TYPE.upper()} Confusion Matrix - Test Set')
    fig.tight_layout()
    plt.savefig(f'{MODEL_TYPE.lower()}_confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Test confusion matrix saved.")
    
    # Clean up
    del y_true, y_pred, preds, cm
    gc.collect()
    
except Exception as e:
    print(f"Error during test evaluation: {e}")

# --- Plot training curves ---
print("Plotting training curves...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title(f'{MODEL_TYPE.upper()} Training and Validation Accuracy & Loss')
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f'{MODEL_TYPE.lower()}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Training curves saved.")
    
    # Show the plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'{MODEL_TYPE.upper()} Training and Validation Accuracy & Loss')
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"Error plotting training curves: {e}")

print("Training completed successfully!")  
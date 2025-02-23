import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
BASE_PATH = './data'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
MODEL_PATH = './models'

def print_progress(message):
    """Print progress with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_model():
    """Create CNN model using ResNet50V2"""
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 scene categories
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def setup_data_generators(df_train, df_val):
    """Setup data generators for training and validation"""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda x: x/255.0
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x/255.0
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=TRAIN_PATH,
        x_col="image_name",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='raw'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=TRAIN_PATH,
        x_col="image_name",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='raw'
    )
    
    return train_generator, val_generator

def save_model(model, history):
    """Save model with version control"""
    os.makedirs(MODEL_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(MODEL_PATH, f'version_{timestamp}')
    os.makedirs(version_path, exist_ok=True)
    
    # Save model
    model.save(os.path.join(version_path, 'model.h5'))
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(version_path, 'training_history.csv'))
    
    # Save model summary
    with open(os.path.join(version_path, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    return version_path

def train():
    print_progress("Starting training process...")
    
    try:
        # Load training data
        print_progress("Loading training data...")
        df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
        print(f"Total training images: {len(df_train)}")
        
        # Split dataset
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
        print(f"Training samples: {len(df_train)}")
        print(f"Validation samples: {len(df_val)}")
        
        # Setup data generators
        print_progress("Setting up data generators...")
        train_generator, val_generator = setup_data_generators(df_train, df_val)
        
        # Create and train model
        print_progress("Creating model...")
        model = create_model()
        model.summary()
        
        print_progress(f"Starting training ({EPOCHS} epochs)...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.ProgbarLogger(count_mode='steps')]
        )
        
        # Save model
        print_progress("Saving model...")
        version_path = save_model(model, history)
        print(f"Model saved in: {version_path}")
        
        return version_path
        
    except Exception as e:
        print_progress(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    train()
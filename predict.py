import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# Configuration
IMG_SIZE = 224
BASE_PATH = './data'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
MODEL_PATH = './models'

# Scene categories mapping
SCENE_CATEGORIES = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

def print_progress(message):
    """Print progress with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def load_model(model_version=None):
    """Load a specific model version or the latest one"""
    try:
        if not os.path.exists(MODEL_PATH):
            print("No models directory found.")
            return None
            
        versions = os.listdir(MODEL_PATH)
        if not versions:
            print("No saved models found.")
            return None
            
        if model_version is None:
            # Load latest version
            latest_version = sorted(versions)[-1]
            model_path = os.path.join(MODEL_PATH, latest_version, 'model.h5')
        else:
            # Load specific version
            if model_version not in versions:
                print(f"Model version {model_version} not found.")
                return None
            model_path = os.path.join(MODEL_PATH, model_version, 'model.h5')
            
        print(f"Loading model from: {model_path}")
        return tf.keras.models.load_model(model_path)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_single_image(model, image_path, show_image=True):
    """Predict category for a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
        
    try:
        img = load_and_preprocess_image(image_path)
        if img is None:
            return None
            
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        scene_category = SCENE_CATEGORIES[predicted_class]
        
        print(f"\nPrediction Results:")
        print(f"Scene: {scene_category}")
        print(f"Confidence: {confidence:.2%}")
        
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(mpimg.imread(image_path))
            plt.title(f"Predicted: {scene_category} ({confidence:.2%})")
            plt.axis('off')
            plt.show()
            
        return scene_category, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def predict_test_set(model, output_file='predictions.csv'):
    """Predict on test set"""
    try:
        # Load test data
        df_test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
        print(f"Total test images: {len(df_test)}")
        
        predictions = []
        for idx, row in df_test.iterrows():
            if idx % 100 == 0:
                print_progress(f"Processing image {idx+1}/{len(df_test)}")
                
            image_path = os.path.join(TRAIN_PATH, row['image_name'])
            img = load_and_preprocess_image(image_path)
            if img is not None:
                pred = model.predict(img, verbose=0)
                predicted_class = np.argmax(pred[0])
                predictions.append({
                    'image_name': row['image_name'],
                    'predicted_scene': SCENE_CATEGORIES[predicted_class],
                    'predicted_class': predicted_class,
                    'confidence': pred[0][predicted_class]
                })
                
        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return pred_df
        
    except Exception as e:
        print(f"Error during test set prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    print("\nScene Classification Predictor")
    print("1. Predict single image")
    print("2. Predict test set")
    choice = input("\nEnter your choice (1-2): ")
    
    # Load model
    model = load_model()
    if model is None:
        exit(1)
    
    if choice == '1':
        image_path = input("Enter path to image: ")
        predict_single_image(model, image_path)
    elif choice == '2':
        output_file = input("Enter output file name (default: predictions.csv): ")
        if not output_file:
            output_file = 'predictions.csv'
        predict_test_set(model, output_file)
# Scene Classification Model

A deep learning model for classifying different types of scenes using TensorFlow and ResNet50V2. The model is designed to classify scenes into 6 different categories.

## Project Structure

```
.
├── data/
│   ├── train/
│   │   └── [image files]
│   ├── train.csv
│   └── test.csv
├── models/
│   └── version_YYYYMMDD_HHMMSS/
│       ├── model.h5
│       ├── training_history.csv
│       └── model_summary.txt
├── train.py
├── predict.py
└── README.md
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install the required packages using:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

## Dataset Structure

### Training Data

- `train.csv`: Contains two columns:
  - `image_name`: Name of the image file
  - `label`: Integer label (0-5) representing the scene category

### Test Data

- `test.csv`: Contains one column:
  - `image_name`: Name of the image file

### Image Directory

- All images should be placed in the `data/train/` directory
- The model expects RGB images (any common format like jpg, png)
- Images will be automatically resized to 224x224 pixels during processing

## Training the Model

The training script (`train.py`) handles:

- Data loading and preprocessing
- Model creation using ResNet50V2
- Training with data augmentation
- Model saving with versioning

To train the model:

```bash
python train.py
```

Training configurations (in `train.py`):

- Image size: 224x224
- Batch size: 32
- Epochs: 10
- Training/Validation split: 80/20
- Data augmentation:
  - Rotation range: 20°
  - Width/Height shift: 20%
  - Horizontal flip

The trained model will be saved in `models/version_YYYYMMDD_HHMMSS/` with:

- Model file (`model.h5`)
- Training history (`training_history.csv`)
- Model architecture summary (`model_summary.txt`)

## Making Predictions

The prediction script (`predict.py`) offers two modes:

1. Single Image Prediction
2. Full Test Set Prediction

To use the predictor:

```bash
python predict.py
```

### Single Image Prediction

- Loads and displays the image
- Shows prediction with confidence score
- Visualizes results using matplotlib

### Test Set Prediction

- Processes all images in test.csv
- Saves predictions to a CSV file with:
  - Image name
  - Predicted class
  - Confidence score

The script automatically uses the latest trained model, but you can modify the code to use a specific version.

## Model Architecture

The model uses ResNet50V2 as the base architecture with additional layers:

- ResNet50V2 (pretrained on ImageNet)
- Global Average Pooling
- Dense layer (256 units, ReLU)
- Dropout (0.5)
- Output layer (6 units, Softmax)

Training parameters:

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

## Error Handling

Both scripts include comprehensive error handling:

- File not found errors
- Image processing errors
- Model loading/saving errors
- Training/prediction errors

All errors are logged with timestamps for easy debugging.

## Best Practices

1. **Data Organization**:

   - Keep all images in the `data/train/` directory
   - Ensure image names match those in CSV files
   - Maintain consistent image formats

2. **Model Versioning**:

   - Each training run creates a new versioned directory
   - Keep track of different model versions for comparison
   - Don't delete old versions until you're sure they're not needed

3. **Resource Management**:
   - Large datasets might require batch processing
   - Consider GPU availability for training
   - Monitor memory usage with large test sets

## Troubleshooting

Common issues and solutions:

1. **Memory Errors**:

   - Reduce batch size in `train.py`
   - Process test set in smaller chunks
   - Close other memory-intensive applications

2. **Image Loading Errors**:

   - Check image file permissions
   - Verify image format compatibility
   - Ensure correct file paths

3. **Model Loading Errors**:
   - Verify model directory exists
   - Check for corrupted model files
   - Ensure compatible TensorFlow versions

## Contributing

Feel free to submit issues and enhancement requests!

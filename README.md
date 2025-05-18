# Face Mask Recognition Neural Network

This project implements a simple neural network in Python to classify images of faces as either **with mask** or **without mask**. It uses NumPy for numerical operations and PIL for image processing, with a custom neural network built from scratch (no deep learning frameworks).

## Features
- Loads and preprocesses face images from specified folders
- Trains a simple feedforward neural network for binary classification
- Evaluates accuracy on a validation set
- Provides a function to classify new images

## Project Structure
```
FaceRecognition/
├── main.py           # Main script for training and testing
├── project/
│   ├── Train/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   └── Validation/
│       ├── WithMask/
│       └── WithoutMask/
```

## Requirements
- Python 3.x
- numpy
- pillow

Install dependencies with:
```bash
pip install numpy pillow
```

## Dataset Structure
- Place training images in `project/Train/WithMask` and `project/Train/WithoutMask`.
- Place validation images in `project/Validation/WithMask` and `project/Validation/WithoutMask`.
- Images should be in formats supported by PIL (e.g., PNG, JPG).

## Usage
1. **Train and Validate**
   Run the main script to train the neural network and evaluate on the validation set:
   ```bash
   python main.py
   ```
   You will see output like:
   ```
   Validation Accuracy: 0.95
   with_mask
   with_mask
   ```

2. **Classify a Single Image**
   Use the `classify_image(image_path, model)` function in `main.py` to classify a new image:
   ```python
   result = classify_image('path/to/image.png', nn)
   print(result)  # Output: 'with_mask' or 'without_mask'
   ```

## Customization
- Adjust `hidden_size`, `learning_rate`, or `steps` in `main.py` to experiment with model performance.
- Change `max_images` in `load_images()` to control how many images are loaded per class.

## Notes
- This is a simple educational project and not intended for production use.
- For best results, use a balanced and diverse dataset.

## License
This project is open source and available under the MIT License.

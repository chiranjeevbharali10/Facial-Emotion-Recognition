import numpy as npsss
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import argparse
import keras.backend as K
import matplotlib.pyplot as plt
import cv2

# Class names (update if your model uses different classes)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image_path, target_size=(48, 48)):
    """
    Preprocess image to match training data format
    Try different sizes: 48x48 (common for emotion detection), 224x224, or 64x64
    """
    # Load and convert to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Try different sizes to find what works
    sizes_to_try = [(48, 48), (64, 64), (224, 224), (128, 128)]
    
    for size in sizes_to_try:
        try:
            # Resize to target size
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            # Convert to numpy array and normalize
            img_array = np.array(img_resized) / 255.0
            
            # Try different channel formats
            if len(img_array.shape) == 3:
                # RGB format
                img_rgb = img_array
                # Grayscale format (single channel)
                img_gray = np.mean(img_array, axis=2, keepdims=True)
                
                # Try both formats
                for format_name, img_formatted in [("RGB", img_rgb), ("Grayscale", img_gray)]:
                    try:
                        # Add batch dimension
                        img_batch = np.expand_dims(img_formatted, axis=0)
                        print(f"Trying {format_name} format with size {size}: {img_batch.shape}")
                        yield img_batch, size, format_name
                    except Exception as e:
                        print(f"Failed with {format_name} format, size {size}: {e}")
                        continue
        except Exception as e:
            print(f"Failed to process size {size}: {e}")
            continue

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    """
    # Clip predictions to prevent log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    
    # Calculate cross entropy
    cross_entropy = -y_true * K.log(y_pred)
    
    # Calculate focal weight
    focal_weight = alpha * K.pow(1 - y_pred, gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * cross_entropy
    
    return K.mean(focal_loss)

def predict_emotion(model_path, image_path):
    """
    Predict emotion for a given image using the trained model
    """
    try:
        # Load the trained model
        print(f"Loading model from {model_path}...")
        
        # Check file extension to determine loading strategy
        file_extension = os.path.splitext(model_path)[1].lower()
        print(f"Model file type: {file_extension}")
        
        if file_extension == '.keras':
            print("Loading .keras model...")
            try:
                # For .keras files, try loading without custom objects first
                model = load_model(model_path, compile=False)
                print("Model loaded successfully with compile=False!")
            except Exception as e:
                print(f"Loading with compile=False failed: {e}")
                print("Trying to load with custom objects...")
                try:
                    model = load_model(model_path, custom_objects={'f1_score': f1_score, 'focal_loss_fixed': focal_loss_fixed}, compile=False)
                    print("Model loaded successfully with custom objects!")
                except Exception as e2:
                    print(f"Loading with custom objects failed: {e2}")
                    print("Trying basic load...")
                    model = load_model(model_path)
                    print("Model loaded successfully with basic load!")
        else:
            # For .h5 files, use the existing logic
            try:
                model = load_model(model_path, custom_objects={'f1_score': f1_score, 'focal_loss_fixed': focal_loss_fixed}, compile=False)
            except Exception as e:
                print(f"Loading with compile=False failed: {e}")
                print("Trying without custom objects...")
                try:
                    model = load_model(model_path, compile=False)
                except Exception as e2:
                    print(f"Loading without custom objects failed: {e2}")
                    print("Trying basic load...")
                    model = load_model(model_path)
        
        # Print model summary for debugging
        print("\nModel Summary:")
        model.summary()
        
        # Get model input shape
        input_shape = model.input_shape
        print(f"\nModel expects input shape: {input_shape}")
        
        # Load original image for display
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Try different preprocessing approaches
        print(f"\nTrying different preprocessing approaches for image: {image_path}")
        
        success = False
        for processed_image, size, format_name in preprocess_image(image_path):
            try:
                print(f"\nTrying prediction with {format_name} format, size {size}")
                print(f"Input shape: {processed_image.shape}")
                
                # Make prediction
                prediction = model.predict(processed_image, verbose=0)
                print("Prediction successful!")
                
                # Get predicted class
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                # Get class name
                predicted_emotion = CLASS_NAMES[predicted_class]
                
                # Print results
                print(f"\nPrediction Results:")
                print(f"Predicted Emotion: {predicted_emotion}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Class Index: {predicted_class}")
                print(f"Used format: {format_name}, size: {size}")
                
                # Print all class probabilities
                print(f"\nAll Class Probabilities:")
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, prediction[0])):
                    print(f"{class_name}: {prob:.4f}")
                
                # Display images side by side
                display_results(original_img, processed_image[0], predicted_emotion, confidence, prediction[0], size, format_name)
                
                success = True
                return predicted_emotion, confidence, prediction[0]
                
            except Exception as e:
                print(f"Failed with {format_name} format, size {size}: {e}")
                continue
        
        if not success:
            print("All preprocessing approaches failed!")
            return None, None, None
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def display_results(original_img, processed_img, predicted_emotion, confidence, all_probabilities, size, format_name):
    """
    Display original image and processed image side by side with prediction results
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display processed image
    ax2.imshow(processed_img)
    ax2.set_title(f'Processed Image ({size[0]}x{size[1]}, {format_name})\nPredicted: {predicted_emotion.upper()}\nConfidence: {confidence:.2%}', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add probability bar chart below
    fig.add_subplot(2, 2, 3)
    plt.bar(CLASS_NAMES, all_probabilities, color=['red', 'orange', 'purple', 'green', 'blue', 'gray', 'yellow'])
    plt.title('Emotion Probabilities', fontsize=12, fontweight='bold')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add confidence text
    plt.figtext(0.5, 0.02, f'Predicted Emotion: {predicted_emotion.upper()} | Confidence: {confidence:.2%} | Format: {format_name} {size[0]}x{size[1]}', 
                ha='center', fontsize=16, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from image using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained .h5 model file')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    # Make prediction
    predict_emotion(args.model, args.image)

# Simple test function for Colab - call this to test different models
def test_models():
    """
    Test different available models to see which one works
    """
    print("Testing available models...")
    
    # List of models to try
    models_to_test = [
        "../Check_points/CNNMODELfinal.keras",
        "../Check_points/best_model.h5", 
        "../Check_points/best_one.h5",
        "../Check_points/resnet50_affectnet.h5",
        "../Check_points/res50(1).h5"
    ]
    
    # List of test images
    test_images = [
        "../Faces/sad.jpg",
        "../Faces/happy.jpg",
        "../Faces/3.jpg"
    ]
    
    print(f"Available models: {models_to_test}")
    print(f"Test images: {test_images}")
    print("\n" + "="*50)
    
    # Test each model with the first image
    for model_path in models_to_test:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing model: {model_path}")
            try:
                result = predict_emotion(model_path, test_images[0])
                if result[0] is not None:
                    print(f"✅ SUCCESS with {model_path}")
                    print(f"   Predicted: {result[0]}, Confidence: {result[1]:.2%}")
                    return model_path, test_images[0]  # Return the working combination
                else:
                    print(f"❌ Failed prediction with {model_path}")
            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    print("\n❌ No working models found!")
    return None, None

# Colab-friendly function - call this directly in Colab
def predict_emotion_colab(model_path="../Check_points/CNNMODELfinal.keras", 
                         image_path="../Faces/sad.jpg"):
    """
    Simple function to call from Colab without command line arguments
    """
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print()
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Available models in Check_points:")
        try:
            import glob
            available_models = glob.glob("../Check_points/*")
            for model in available_models:
                print(f"  - {model}")
        except:
            print("Could not list available models")
        return None, None, None
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Available images in Faces:")
        try:
            import glob
            available_images = glob.glob("../Faces/*")
            for img in available_images:
                print(f"  - {img}")
        except:
            print("Could not list available images")
        return None, None, None
    
    return predict_emotion(model_path, image_path)

if __name__ == "__main__":
    # If no command line arguments, use default values
    import sys
    if len(sys.argv) == 1:
        # Default values - update these paths as needed
        model_path = "../Check_points/CNNMODELfinal.keras"  # Using .keras model (relative to TESTING dir)
        image_path = "../Faces/sad.jpg"    # Update to your image path (relative to TESTING dir)
        
        print(f"Using default paths:")
        print(f"Model: {model_path}")
        print(f"Image: {image_path}")
        print()
        
        # Check if files exist before proceeding
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            print("Available models in Check_points:")
            try:
                import glob
                available_models = glob.glob("../Check_points/*")
                for model in available_models:
                    print(f"  - {model}")
            except:
                print("Could not list available models")
            sys.exit(1)
        
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            print("Available images in Faces:")
            try:
                import glob
                available_images = glob.glob("../Faces/*")
                for img in available_images:
                    print(f"  - {img}")
            except:
                print("Could not list available images")
            sys.exit(1)
        
        predict_emotion(model_path, image_path)
    else:
        main() 
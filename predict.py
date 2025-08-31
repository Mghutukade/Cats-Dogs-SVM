import cv2
import sys
from features import extract_hog_features
from utils import load_model

MODEL_PATH = "models/svm_cats_dogs.joblib"

def predict_image(image_path):
    # Load model
    model = load_model(MODEL_PATH)

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image not found: {image_path}")
        return None

    img = cv2.resize(img, (64, 64))
    features = extract_hog_features([img])

    # Predict
    prediction = model.predict(features)
    return "Cat" if prediction[0] == 0 else "Dog"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        if result:
            print(f"Predicted: {result}")

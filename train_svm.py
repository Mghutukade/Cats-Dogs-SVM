import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import split_data, save_model
from dataset import load_data
from features import extract_hog_features



DATA_DIR = "../data/train"
MODEL_PATH = "models/svm_cats_dogs.joblib"

def train_svm():
    print("Loading dataset...")
    X, y = load_data(DATA_DIR, img_size=64)

    print("Extracting HOG features...")
    X = extract_hog_features(X)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training SVM model...")
    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    save_model(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_svm()

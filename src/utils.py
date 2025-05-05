import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

LABELS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def load_and_compile_model(model_path, learning_rate=0.001, loss='categorical_crossentropy', metrics=['accuracy']):
    """
    Loads a Keras model from the given path and compiles it with the specified parameters.

    Parameters:
    - model_path: path to the .h5 model file
    - learning_rate: optimizer learning rate
    - loss: loss function
    - metrics: list of metrics to use

    Returns:
    - compiled Keras model
    """
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model

def preprocess_image(image_path, target_size=(299, 299)):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image_label(image_path, model, labels, target_size=(299, 299)):
    """
    Returns the label with the highest probability, and full probability distribution.
    """
    img = preprocess_image(image_path, target_size)
    predictions = model.predict(img, verbose=0)[0]
    prediction_dict = {labels[i]: float(predictions[i]) for i in range(len(labels))}
    max_label = max(prediction_dict, key=prediction_dict.get)
    return max_label, prediction_dict

from tqdm import tqdm
def generate_predictions(image_paths, true_labels, model, labels=LABELS, target_size=(299, 299)):
    """
    Generates predictions and returns both class names and numeric indices.

    Returns:
        y_true (list): Original string labels
        y_pred (list): Predicted string labels
        y_true_numeric (list): True labels as indices
        y_pred_numeric (list): Predicted labels as indices
    """
    assert len(image_paths) == len(true_labels), "Mismatch in image paths and true labels"

    y_true = []
    y_pred = []

    for img_path, true_label in tqdm(zip(image_paths, true_labels), total=len(image_paths), desc="Generating predictions"):
        predicted_label, _ = predict_image_label(img_path, model, labels, target_size)
        y_true.append(true_label)
        y_pred.append(predicted_label)

    try:
        y_true_numeric = [labels.index(label) for label in y_true]
        y_pred_numeric = [labels.index(label) for label in y_pred]
    except ValueError as e:
        raise ValueError(f"Label not found in label list: {e}")

    return y_true, y_pred, y_true_numeric, y_pred_numeric

def evaluate_classification_results(y_true, y_pred, output_dir, labels=LABELS):
    """
    Evaluate classification results and save accuracy metrics, classification report, and confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "Total Images": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted")
    }

    # Save metrics
    with open(os.path.join(output_dir, "metrics_post.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    with open(os.path.join(output_dir, "classification_report_post.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=labels))
    # Save confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.9)  # lighter fill
    # Add colorbar
    fig.colorbar(cax)
    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    # Remove inner grid lines (keep only external borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Draw outer rectangle manually
    ax.plot([-0.5, 3.5, 3.5, -0.5, -0.5], [-0.5, -0.5, 3.5, 3.5, -0.5], color='black', linewidth=1.5)
    # Annotate each cell
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]),
                va='center', ha='center', color='black', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_post.png"))
    plt.show()

def load_image_paths_and_labels(folder_paths, extension=".jpeg"):
    """
    Load image file paths and labels from one or more folders where subfolders are labels.
    """
    image_paths = []
    true_labels = []
    for folder_path in folder_paths:
        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    if img_name.lower().endswith(extension):
                        image_paths.append(os.path.join(label_path, img_name))
                        true_labels.append(label)
    return image_paths, true_labels

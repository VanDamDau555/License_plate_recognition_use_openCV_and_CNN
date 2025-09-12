import tensorflow as tf
import numpy as np
import cv2
import string
import argparse
from models.Mymodels import CNN_model 

# Tạo mapping 0-9 và a-z
CLASS_NAMES = [str(i) for i in range(10)] + list(string.ascii_lowercase)

def load_model(weight_path, input_shape=(64, 64, 1), num_classes=36):  
    model = CNN_model(inp_shape=input_shape, num_classes=num_classes)
    model.model.load_weights(weight_path)  # Load weights
    return model.model

def preprocess_image(img_path, input_shape=(64, 64, 1)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Can't read image: {img_path}")

    # Resize
    img = cv2.resize(img, (input_shape[0], input_shape[1]))

    # Convert RGB to Gray 
    if input_shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1) 

    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def predict(model, img_path, input_shape=(64, 64, 1)):
    """Predict class from image"""
    img = preprocess_image(img_path, input_shape)
    preds = model.predict(img)
    class_id = np.argmax(preds, axis=1)[0]
    return CLASS_NAMES[class_id], preds[0][class_id]

def main(args):
    model = load_model(args.weights, input_shape=(args.img_height, args.img_width, args.img_channels))
    label, conf = predict(model, args.image, input_shape=(args.img_height, args.img_width, args.img_channels))
    print(f"Predict: {label} (confidence={conf:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./models/LandD_last_model.h5", help="Path to .h5 weights file")
    parser.add_argument("--image", type=str, required=True, help="Give to path of input image")
    parser.add_argument("--img_height", type=int, default=64, help="Height of input image")
    parser.add_argument("--img_width", type=int, default=64, help="Width of input image")
    parser.add_argument("--img_channels", type=int, default=1, help="(1=Gray, 3=RGB)")
    args = parser.parse_args()

    main(args)

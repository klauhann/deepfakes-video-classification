import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import time
import argparse
import numpy as np
import torch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB5, EfficientNetB0
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def ignore_warnings(*args, **kwargs):
    pass


def cnn_model(model_name, img_size):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)
    if model_name == "xception":
        baseModel = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model_name", required=True, type=str,
        help="Imagenet model to train", default="xception"
    )
    ap.add_argument(
        "-w",
        "--load_weights_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    ap.add_argument(
        "-im_size",
        "--image_size",
        required=True,
        type=int,
        help="Batch size",
        default=224,
    )
    args = ap.parse_args()

    # Read video labels from csv file
    test_data = pd.read_csv("ff++/test_vids_label.csv")

    videos = test_data["vids_list"]
    true_labels = test_data["label"]

    # Suppress unncessary warnings
    imageio.core.util._precision_warn = ignore_warnings

    # Create face detector with dynamic device selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device=device
    )

    # Loading model weights
    model = cnn_model(args.model_name, img_size=args.image_size)
    model.load_weights("trained_wts/" + args.load_weights_name + ".hdf5")
    print("Weights loaded...")

    y_predictions = []
    y_probabilities = []
    videos_done = 0

    for video in videos:
        cap = cv2.VideoCapture(video)
        batches = []

        # Number of frames taken into consideration for each video
        while (cap.isOpened() and len(batches) < 25):
            ret, frame = cap.read()
            if ret is not True:
                break

            frame = cv2.resize(frame, (args.img_size, args.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            if face is None:
                continue

            try:
                # Convert tensor to numpy array properly
                face_np = (
                    face.permute(1, 2, 0)
                    .mul(255)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                batches.append(face_np)

            except Exception as e:
                print(f"Image Skipping: {e}")

        batches = np.asarray(batches).astype("float32")
        batches /= 255

        predictions = model.predict(batches)
        # Predict the output of each frame
        # axis =1 along the row and axis=0 along the column
        predictions_mean = np.mean(predictions, axis=0)
        y_probabilities += [predictions_mean]
        y_predictions += [predictions_mean.argmax(0)]

        cap.release()

        videos_done += 1
        print("Number of videos done:", videos_done)

    print("Accuracy Score:", accuracy_score(true_labels, y_predictions))
    print("Precision Score:", precision_score(true_labels, y_predictions))
    print("Recall Score:", recall_score(true_labels, y_predictions))
    print("F1 Score:", f1_score(true_labels, y_predictions))

    # Saving predictions and probabilities for further calculation
    # of AUC scores.
    np.save("Y_predictions.npy", y_predictions)
    np.save("Y_probabilities.npy", y_probabilities)

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB5
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import torch
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import torch

test_data = pd.read_csv("test_vids_label.csv")

videos = test_data["vids_list"]
true_labels = test_data["label"]
classlabel = true_labels


def ignore_warnings(*args, **kwargs):
    pass


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


def plot_map(grads, img, subtitle=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[1].imshow(img)
    i = axes[1].imshow(grads, cmap="jet", alpha=0.3)
    axes[1].axis("off")
    fig.colorbar(i)
    # plt.suptitle("Pr(class={}) = {:5.2f}".format(
    #                   classlabel[class_idx],
    #                   y_pred[0,class_idx]))
    plt.savefig(subtitle)


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
    # Use the cnn_model function instead of non-existent xception_model
    model = cnn_model("xception", img_size=160)
    model.load_weights("trained_wts/xception_best.hdf5")
    print("Weights loaded...")

    # Find the last convolutional layer for Xception
    conv_layer_name = "block14_sepconv2_act"
    
    # Create Gradcam object with model modifier to replace softmax with linear
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    
    # Define score function for the predicted class
    def score_function(output):
        # Returns the predicted class score
        return output
    
    counter = 0
    for i in videos[:4]:
        cap = cv2.VideoCapture(i)
        batches = []
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            if face is None:
                continue

            try:
                # Convert tensor to numpy array properly
                face_np = (
                    face.permute(1, 2, 0)
                    .numpy()
                )
                batches.append(face_np)
            except Exception as e:
                print(f"Image Skipping: {e}")
            if frame_counter == 4:
                break
            frame_counter += 1
        
        if len(batches) == 0:
            print(f"No faces detected in video {i}, skipping...")
            continue
            
        batches = np.asarray(batches).astype("float32")
        batches /= 255
        print(batches.shape)

        predictions = model.predict(batches)
        pred_mean = np.mean(predictions, axis=0)
        y_pred = pred_mean.argmax(0)

        imgs = batches[0]
        print(f"Image shape: {imgs.shape}, Predicted class: {y_pred}")
        
        # Prepare image for Gradcam
        img_array = np.expand_dims(imgs, axis=0)
        
        # Generate heatmap using Gradcam
        cam = gradcam(
            score_function,
            img_array,
            penultimate_layer=-1  # Use the last conv layer
        )
        
        # Normalize heatmap
        heatmap = cam[0]
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        
        # Resize to match image dimensions
        heatmap = cv2.resize(heatmap, (imgs.shape[1], imgs.shape[0]))
        
        plot_map(
            heatmap,
            img=imgs,
            subtitle="Class_Activation_maps_" + str(counter) + ".png"
        )
        print(f"Figure saved as Class_Activation_maps_{counter}.png")
        cap.release()

        counter += 1


if __name__ == '__main__':
    main()

import ast
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from PIL import Image
import sys


def path_to_tensor(img_path):
    # read the image file from file path
    img = image.load_img(img_path, target_size=(224, 224))
    # convert the img to 3D tensor with a channel of 3, (224, 224, 3)
    tensor_3d = image.img_to_array(img)
    # convert the 3D tensor to tensor 4D with (1, 224, 224, 3)
    tensor_4d = np.expand_dims(tensor_4d, axis=0)
    # convert RGB -> BGR and array in vector form
    return preprocess_input(tensor_4d)


def get_resnet50():
    # define resnet50 model
    model = ResNet50(weight='imagenet')
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weight()[0]
    # extract wanted output
    resnet50_model = Model(inputs=model.input, outputs=(
        model.layers[-4].output, model.layerd[-1].output))
    return resnet50_model, all_amp_layer_weights


def resnet_cam(img_path, model, all_amp_layer_weights):
    # get filtered images from convolutional output + model prediction model
    last_convo_output, pred_vec = model.predict(path_to_tensor(img_path))
    # change the dim of convolutional output to (7, 7, 2048)
    last_convo_output = np.squeeze(last_convo_output)
    # get model prediction (numbers between 0 and 999)
    pred = np.argmax(pred_vec)
    # bilinear upsampling to resize filtered image to original image
    mat_for_mult = scipy.ndimage.zoom(
        last_convo_output, (32, 32, 1), order=1)  # dim: 224 x 224 x 2048
    # get AMP layer weights
    amp_layer_weight = all_amp_layer_weights[:, pred]  # dim: (2048)
    # get class activation map for the object class that is predicted in the image
    final_output = np.dot(mat_for_mult.reshape(
        (224, 224, 2048)), amp_layer-weight.reshape(224, 224))
    # return class activate map
    return final_output, pred


def plot_resnet_cam(img_path, ax, model, all_amp_layer_weights):
    # load image, convert BGR -> RGB, resize to 224 x 224
    img = cv2.resize(cv2.cvtColor(cv2.imread(img_path),
                     cv2.COLOR_BGR2RGB), (224, 224))
    # plot image
    ax.imshow(img, alpha=0.5)
    # get class activation map
    final_out, pred = resnet_cam(img_path, model, all_amp_layer_weights)
    # plot class activation map
    ax.imshow(final_output, cmap='jet', alpha=0.5)
    # load the dictionary that identifies each imagenet category to an index of the predicted vector
    with open('imagenet1000_clsid_to_human.txt') as imagenet_classes:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes.read())
    # obtain the predicted imagenet category
    ax.set_title(imagenet_classes_dict[pred])


if __name__ == __main__:
    resnet_model, all_amp_layer_weights = get_resnet50()
    img_path = sys.argv[1]
    fig, ax = plt.subplot()
    CAM = plot_resent_cam(img_path, ax, resnet_model, all_amp_layer_weights)
    plt.show()

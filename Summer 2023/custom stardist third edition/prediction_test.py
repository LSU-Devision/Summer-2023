from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import normalize
from glob import glob
from skimage.color import rgb2gray
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D
from tifffile import imread
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# displays original pixels without any smoothing or interpolation
matplotlib.rcParams["image.interpolation"] = "none"

# set a random seed for reproducibility
np.random.seed(42)

# generates a random color map for labeled regions
lbl_cmap = random_label_cmap()

# directory containing images for prediction
image_dir = "prediction"

# sorts files by name
file_names = sorted(os.listdir(image_dir))

# selects all .tif images
X = sorted(glob("prediction/*.tif"))

# read images from X
X = list(map(imread,X))

# transfroms the images to grayscale
X = [rgb2gray(img) for img in X]

# determine the number of channels in the images
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# define the axes for normalization
axis_norm = (0,1)

# load the specific model you created from the directory of your models
model = StarDist2D(None, name="customModel_15_epochs_100", basedir="models/datasize_15")

# function for prediction
def prediction(model, i, show_dist=True):

    # normalize the image for better prediction quality
    img = normalize(X[i], 1, 99.8, axis=axis_norm)

    # uses CPU for prediction - can run into memory problems otherwise
    with tf.device("/cpu:0"):
        labels, details = model.predict_instances(img)

    # count the number of unique labels (subtract 1 to exclude the background label 0)
    num_objects = len(np.unique(labels)) - 1

    # get the file name for the current image
    file_name = file_names[i]

    # get the actual count from the file name
    # for this to work, your files need to be labeled in this way:
    # ###count.jpg
    actual_count = int(file_name.split("count")[0])

    # initialize a plot of specified size
    plt.figure(figsize=(13, 10))

    # checks if the image is grayscale or color for visualization
    img_show = img if img.ndim == 2 else img[..., 0]

    # extract details for drawing polygons
    coord, points, prob = details["coord"], details["points"], details["prob"]
    
    # display the original image
    plt.subplot(121); plt.imshow(img_show, cmap="gray"); plt.axis("off")

    # saves axis settings for overlay
    a = plt.axis()

    # overlay polygons representing object boundaries
    _draw_polygons(coord, points, prob, show_dist=show_dist)

    # reset axis to the saved settings
    plt.axis(a)

    # adds title
    plt.title(f"Predicted Objects: {num_objects}\nActual Objects: {actual_count}", fontsize=16)
    
    # display the segmented labels over the original image
    plt.subplot(122); plt.imshow(img_show, cmap="gray"); plt.axis("off")
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)

    # adjust layout
    plt.tight_layout()
    
    # save the figure to your specified directory
    plt.savefig(f"prediction/prediction_test{i}.png", dpi=500)
    plt.close()

# starts the process of predicting
# will predict for each image in X
for i in range(len(X)):
    prediction(model, i)
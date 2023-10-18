# necessary imports
from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import Path, normalize
from glob import glob
from stardist import fill_label_holes, calculate_extents
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from tifffile import imread
from tqdm import tqdm
import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import threading
import time

# X_filenames represents the raw images
X_filenames = sorted(glob("images/*.tif"))

# function to parse arguments given in command line
def parse_args():
    parser = argparse.ArgumentParser(description="Here is the help section for the optional commands.")
    parser.add_argument("--total_data", type=int, default=len(X_filenames), help="Sets the total amount of data. Default: total amount of images in the images folder.")
    parser.add_argument("--dataset_size", type=int, default=int(.75*len(X_filenames)), 
        help="Sets the size of the dataset to be used. Cannot be equal to total_data as there would be no testing data and the program will not work. Default: .75 of the total_data.")
    parser.add_argument("--rays", type=int, default=32, help="Sets the number of Rays. Default: 32.")
    parser.add_argument("--train_split", type=float, default=0.85, help="Sets the percent to split training/validation data. Default: .85.")
    parser.add_argument("--testing_size", type=int, default=1, help="Sets the number of testing images. Default: 1 to ensure the program works.")
    parser.add_argument("--epochs", type=int, nargs='+', default=[10], 
        help="Sets the number of epochs. Accepts a number or list of numbers. E.g. --epochs 10 50 100 300. Default: 10.")
    parser.add_argument("--model_name", type=str, default="customModel", help="Sets the name of the model. Accepts a string. Default: customModel")
    return parser.parse_args()

# main function
def main(args):

    # Y represents the masks
    Y = sorted(glob("masks/*.tif"))

    # assertion check ensures that each image has a matching mask with the same filename
    assert all(Path(x).name==Path(y).name for x,y in zip(X_filenames,Y))

    # read image files and store them in a list named X
    X = list(map(imread, X_filenames))

    # read image files and store them in a list named Y
    Y = list(map(imread, Y))
    # ensures that any random operations are reproducible across all runs
    np.random.seed(42)

    # print statements
    print(f"total amount of data: {args.total_data}")
    print(f"dataset_size: {args.dataset_size}")
    print(f"testing size: {args.testing_size}")
    print(f"rays: {args.rays}")
    print(f"training/validation split: {args.train_split}")
    print(f"epochs: {args.epochs}")
    print("total number of images:", len(X))
    print("total number of masks:", len(Y))

    # function for grayscale
    def rgb_to_gray(img):
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    # function for grayscale
    def ensure_grayscale(X):
        return [rgb_to_gray(x) if x.ndim == 3 else x for x in X]

    # ensures that the input images are grayscale
    X = ensure_grayscale(X)

    # n_channels are set to 1 indicating that the input images are grayscale
    # this is a single channel as opposed to RGB images which have 3 channels
    n_channel = 1

    # normalization of images
    axis_norm = (0,1)
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]

    # applied to each mask in Y, ensuring that any labeled regions are solid without internal holes
    Y = [y.astype(np.int32) for y in Y]
    Y = [fill_label_holes(y) for y in Y]

    # function to pad images to ensure that the images meet the minimum requirement of 256x256 pixels
    def pad_image(img, target_shape=(256, 256)):
        pads = [(0, max(0, target_shape[i]-img.shape[i])) if i < 2 else (0, 0) for i in range(img.ndim)]
        return np.pad(img, pads, mode='constant')

    # padding images
    X = [pad_image(x) for x in X]
    Y = [pad_image(y) for y in Y]
    Y = [y.astype(np.int32) for y in Y]
    # number of rays to use for the non-maximum suppression in the StarDist model
    # 32 is a great starting point
    n_rays = args.rays

    # parameter in StarDist that specifies the step size for the patches extracted from
    # the training images
    grid = (2,2)

    # configuration object with model parameters to be used in initilization/training
    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        n_channel_in = n_channel,
    )

    # function for random flips/rotations of the data
    def random_fliprot(img, mask): 
        assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(perm) 
        for ax in axes: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    # function to give data random intensity
    def random_intensity_change(img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img

    # this is the augmenter that will go into the training of the model
    def augmenter(x, y):
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        sig = 0.02*np.random.uniform(0,1)
        x = x + sig*np.random.normal(0,1,x.shape)
        return x, y

    # should be set to the same number as the seed at the beginning of the script
    rng = np.random.default_rng(42)

    # size of dataset to train on
    dataset_size = args.dataset_size

    # this is only relevant when you want different versions
    # version = 1

    # this should only be used if you want different versions
    # rng = np.random.default_rng(version)

    # amount of total data: change this to suit your needs
    total_data = args.total_data

    # selects 36 random unique images for testing from the total 180 images
    # change 180 to amount of data you have: replace is set to false to ensure
    # that the program selects unique images for the testing set
    test_indices = rng.choice(total_data, size=args.testing_size, replace=False)

    # get your test datasets
    X_test, Y_test = [X[i] for i in test_indices], [Y[i] for i in test_indices]

    # remaining images are the ones that aren't in the test set
    all_remaining_indices = list(set(range(total_data)) - set(test_indices))

    # select subset from the remaining based on dataset_size
    selected_indices = rng.choice(all_remaining_indices, size=dataset_size, replace=False)

    # determine sizes for training and validation
    # % of the dataset_size
    n_train_split = args.train_split
    n_train = int(n_train_split * dataset_size)

    # split the selected data into training and validation
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    # get your training and validation datasets
    X_train, Y_train = [X[i] for i in train_indices], [Y[i] for i in train_indices]
    X_val, Y_val = [X[i] for i in val_indices], [Y[i] for i in val_indices]

    # prints amount of images in training, validation, and testing to verify
    print(f"training set size: {len(X_train)}")
    print(f"validation set size: {len(X_val)}")

    # where the model will be saved
    base_dir = "models"

    # make a new directory for the dataset size
    dataset_dir = os.path.join(base_dir, f'datasize_{dataset_size}')
    os.makedirs(dataset_dir, exist_ok=True)

    # function to save training, validation, and testing data in a grid to a PNG file
    def save_images_to_file(images, filename, title):
        # specify the dimensions of the subplot grid
        n = len(images)
        cols = int(math.sqrt(n))  # assuming you want a square grid, change this as per your requirements
        rows = int(math.ceil(n / cols))

        # create a new figure with specified size
        fig = plt.figure(figsize=(20, 20))  # adjust as needed

        # set title
        plt.title(title, fontsize=40)  # adjust font size as needed

        # iterate over each image and add it to the subplot
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.imshow(images[i], cmap='gray')  # using gray colormap as these are grayscale images
            ax.axis('off')  # to remove axis

        # adjust layout and save the figure
        fig.tight_layout()  # adjust layout so labels do not overlap
        fig.savefig(filename, dpi=600)

    # this section saves the training, validation, and testing data in separate images to
    # see the data the program selected using the function from above
    # saving the training images
    training_filename = os.path.join(dataset_dir, 'training_images.png')  # define the path and name for your image
    save_images_to_file(X_train, training_filename, "Training Images")

    # saving the validation images
    validation_filename = os.path.join(dataset_dir, 'validation_images.png')  # define the path and name for your image
    save_images_to_file(X_val, validation_filename, "Validation Images")

    # saving the testing images
    testing_filename = os.path.join(dataset_dir, 'testing_images.png')  # define the path and name for your image
    save_images_to_file(X_test, testing_filename, "Testing Images")

    # function to evaluate and save csv files
    def evaluate_and_save(model, X_data, Y_data, data_type='validation'):

        # prediction
        Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(X_data)]
        
        # evaluation
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(Y_data, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
        
        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
        counts = ('fp', 'tp', 'fn')
        
        for m in metrics:
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in counts:
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        
        # save figure
        figure_filename = os.path.join(model.basedir, model.name, f"{data_type}_plots.png")
        fig.savefig(figure_filename, dpi=300)
        
        # save CSV
        filename = os.path.join(model.basedir, model.name, f'{data_type}_stats.csv')
        fieldnames = list(stats[0]._asdict().keys())
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in stats:
                writer.writerow(entry._asdict())

        return stats

    # number of epochs
    epochs = args.epochs

    # main training loop
    for i in epochs:

        # naming the model
        model_name = args.model_name + "_" + str(args.dataset_size) + '_epochs_' + str(i)

        # naming the model - this is used if there will multiple versions of the same training
        # this is more advanced
        # model_name = 'customStardist_' + str(dataset_size) + '_v' + str(version) + '_epochs_' + str(i)

        # instantiate the model with custom parameters
        model = StarDist2D(conf, name=model_name, basedir=dataset_dir)

        # calculates the average size of labeled objects in mask images
        median_size = calculate_extents(list(Y_train), np.median)

        # refers to how much the network can "see" the image in a single pass
        fov = np.array(model._axes_tile_overlap('YX'))

        # printing median size and fov
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")

        # this is to warn the user that the median object size is larger than the fov
        # which can cause the network to struggle to detect the objects properly
        # this can lead to partial segmentations or missed detections
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")

        # epochs based on where i is in the list of epochs
        epochs = i

        # code to train the model based on the data given
        model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=epochs)

        # optimizing thresholds for validation data
        model.optimize_thresholds(X_val, Y_val)

        # evaluation of validation data
        stats_val = evaluate_and_save(model, X_val, Y_val, 'validation')

        # evaluation of testing data
        stats_test = evaluate_and_save(model, X_test, Y_test, 'test')
    print("Training is complete.")

if __name__ == "__main__":
    # loading arguments
    args = parse_args()
    main(args)

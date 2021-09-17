# Project - Zero shot learning using k-means clustering 
# Data Set - Animals with Attributes 2 (Refer to the repo. for additional details)
import os
import sys
import argparse
import cv2
import torch
import random
import struct
import train 
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils import data
from resnet import ResNet101
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torchvision.transforms import transforms
from sklearn.metrics import silhouette_samples, silhouette_score
from AnimalDataset import AnimalDataset, colors_per_class, colors_per_class_train, colors_per_class_test

# Global path to access the data files 1. Using Windows use path value as 'tmp/' || 2. Using Ubuntu/Linux & MacOS use path value as '/tmp/'
if os.name =='nt':
    PATH = './'
else:
    PATH = './'

def k_means_cluster(labels_features_dict,num_clusters):
    #global centroid 
    centroid = None
    
    # List to store the class labels
    labels = list()

    for key in labels_features_dict:
        kmeans = KMeans(n_clusters = num_clusters, random_state=0).fit(labels_features_dict[key])
        if centroid is None:
            centroid = kmeans.cluster_centers_
        else:
            centroid = np.concatenate((centroid,kmeans.cluster_centers_))
        for i in range(kmeans.cluster_centers_.shape[0]):
            labels.append(key)
        #print(labels_features_dict)
    #print(labels)
    #print("Cluster Labels - ", sorted(np.unique(kmeans.labels_)))
    #print(centroid)
    return labels, centroid

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# Function to train the network
def train_network(num_features,batch_size):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    #We build the model here.
    #The model is based on the number of features in the dataset
    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 3}
    train_process_steps = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((224,224)), # ImageNet standard
    transforms.ToTensor()
    ])
    eval_interval = 5
    
    # Epochs value for the model
    num_epochs = 20
    model = torchvision.models.resnet50(pretrained=True)
    model_features = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Sequential(nn.BatchNorm1d(model_features), nn.ReLU(), nn.Dropout(0.25), nn.Linear(model_features, num_features)))
    model = model.to(device)
    
    #To load the train dataset
    train_dataset = AnimalDataset('trainclasses.txt', train_process_steps, PATH)
    trainloader = data.DataLoader(train_dataset, **train_params)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    total_steps = len(trainloader)
    
    
    # Model Training starts
    # To check the condition if the model exists or not.
    if not os.path.exists('models/model_020'):
        if not os.path.exists('models'):
            os.mkdir('models')
        for epoch in range(num_epochs):
          for i, (images, features, img_names, indexes) in enumerate(trainloader):
            
            # Batchnorm1D can't handle batch size of 1
            if images.shape[0] < 2:
              break
            images = images.to(device)
            features = features.to(device).float()
            
            # Toggle training flag
            model.train()
            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)
            loss = criterion(sigmoid_outputs, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
              curr_iter = epoch * len(trainloader) + i
              print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
              sys.stdout.flush()
            
            # The model and optimizer files will be saved under the path 'models/model_name or models/optimizer_name'
            torch.save(model.state_dict(), 'models/{}'.format('model_020'))
            torch.save(optimizer.state_dict(), 'models/{}'.format('Adam_020'))


        # Do some evaluations (Not required for now)
        # if (epoch + 1) % eval_interval == 0:
        #   print('Evaluating:')
        #   curr_acc = evaluate(model, test_loader)
        #   print('Epoch [{}/{}] Approx. training accuracy: {}'.format(epoch+1, num_epochs, curr_acc))
        
        # If the model path does not exist create the model directory
        if not os.path.exists('models'):
            os.mkdir('models')
        
        # Steps to save model and optimizer
        torch.save(model.state_dict(), 'models/{}'.format('model_020'))
        torch.save(optimizer.state_dict(), 'models/{}'.format('Adam_020'))

    else:
        print("PATH models/model_020 exists! no need to train. Use pretrained model")
        
        # To load the already trained model file.
        # Remove map_location = 'cpu' when executing on GPU in model.load_state_dict(torch.load('models/model_020')).
        model.load_state_dict(torch.load('models/model_020'))

        # Evaluation Step
        model.eval()
    return model

# Function to extract the features of the image data
def get_features(batch, num_images, model):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 3}
    # read the dataset and initialize the data loader
    dataloader = []
    train_process_steps = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((224,224)), # ImageNet standard
    transforms.ToTensor()
    ])
    test_process_steps = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

    # To load the train data set
    dataset_train = AnimalDataset('trainclasses.txt', train_process_steps, PATH)
    
    # To load the test data set
    dataset_test = AnimalDataset('testclasses.txt', test_process_steps, PATH)
    
    dataloader.append(torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True))
    dataloader.append(torch.utils.data.DataLoader(dataset_test, batch_size=batch, shuffle=True))
    
    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    labels_features_dict = dict()
    # images_set = []
    # images_set = dataloader[0].dataset.img_names
    # images_set.append(dataloader[1].dataset.img_names)

    for dataloaderum in dataloader:
        for (images, feat, img_names, indexes) in tqdm(dataloaderum, desc='Running the model inference'):

            labels+=img_names
            with torch.no_grad():
                output = model.forward(images.to(device))
            
            # Extracting the features
            current_features = output.cpu().numpy()

            # To add the new feature with the existing features
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
            
            # Step to add a threshold value to the features to form compact and close clusters
            # Try with different threshold values
            features[features > 0.25] = 1
            features[features < 0.25] = 0

    # To extract class names from the image_path " tmp/data/Animals_with_Attributes2/JPEGImages/(class_name)/(image_name.ext) "
    for label in zip(labels,features):
        try:
            if labels_features_dict is None:
                labels_features_dict[label[0]] = np.array(label[1])
            else:
                labels_features_dict[label[0]] = np.concatenate((np.array(labels_features_dict[label[0]]), np.expand_dims(np.array(label[1]),axis=0)))
        except KeyError:
            labels_features_dict[label[0]] = None
            labels_features_dict[label[0]] = np.array(np.array(np.expand_dims(label[1],axis=0)))
    #print(labels_features_dict)
    return features, labels, image_paths, labels_features_dict

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution value_range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image

def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    #plt.show()

# Function to calculate the euclidean distance between train and test
def euclidean_distance(tx_c, ty_c, labels):

    train_classes_x = dict()
    train_classes_y = dict()
    test_classes_x = dict()
    test_classes_y = dict()
    minimum_distance_train_test = dict()

    for train_label in colors_per_class_train:
        indices = [i for i, l in enumerate(labels) if l == train_label]
        x_train = np.take(tx_c, indices)
        y_train = np.take(ty_c, indices)
        #train_classes_x_y[train_label] = [x_train , y_train]
        train_classes_x[train_label] = x_train
        train_classes_y[train_label] = y_train


    for test_label in colors_per_class_test:
        indices = [i for i, l in enumerate(labels) if l == test_label]
        x_test = np.take(tx_c, indices)
        y_test = np.take(ty_c, indices)
        test_classes_x[test_label] = x_test
        test_classes_y[test_label] = y_test
        # print("Test class x & y coordinates :" ,test_label, x_test, y_test)
    
    for (test_label_x), (test_label_y) in zip(test_classes_x.keys(), test_classes_y.keys()):
        for (train_label_x), (train_label_y) in zip(train_classes_x.keys(), train_classes_y.keys()):
            distance = np.sqrt((train_classes_x[train_label_x] - test_classes_x[test_label_x])**2 + (train_classes_y[train_label_y] - test_classes_y[test_label_y])**2)
            #distance_norm = np.linalg.norm(distance)
            #normal_array = distance/distance_norm
            
            """ norm = np.linalg.norm(an_array)
            normal_array = an_array/norm """
            
        # Calculates the minimum distance between two set of points. 
        MinDist = np.min(distance)

        # Store the distance in a dictionary 
        minimum_distance_train_test[train_label_x, test_label_x] = MinDist
    #print(normal_array)
    print("\nActual Min. Distance :", minimum_distance_train_test)
    print("\nDistance", distance)

    # distance = np.sqrt((x_train-x_test)**2 + (y_train-y_test)**2)
    #Minimum_Distance = min(distance)
    #minimum_distance_train_test[train_label, test_label] = Minimum_Distance
    #print(minimum_distance_train_test)

    # Maximum_Distance = max(distance)
    # Average_Distance = 0 if len(distance) == 0 else sum(distance)/len(distance)


    """ 
    1. Store Train classes X & Y points in a 2D Array
    2. Store Test classes X & Y points in a 2D Array 
    3. Access the values in the 2D Array and calculate the distances of one train cluster with one test data point
    4. Calculate the minimum distance from the relation of one train and test class. Add the value to the new matrix.    
    """

    # Create a minimum_distance_matrix
    # minimum_distance_matrix = np.empty((len(colors_per_class_test), len(colors_per_class_train)), float)
    print("Length colors_per_class_test:", len(colors_per_class_test),"\nLength colors_per_class_train:", len(colors_per_class_train))
    # print("Min Distance Matrix:", minimum_distance_matrix)
    
    """ print("Calculating the Euclidean distance between classes : \n")
    print("<===============================================================\n")
    #Print Distances
    print("Distance between data points of train and test : " , distance)
    #max
    print("\nMaximum Distance :" , Maximum_Distance)
    #min
    print("\nMinimum Distance :", Minimum_Distance)
    #avg
    print("\nAverage Distance :", Average_Distance)
    print("===============================================================>\n") """
    
# Add a function to create a confusion matrix
def confusion_matrix(train_classes, test_classes, labels):

    confusion_mat = np.array(confusion_matrix(train_classes, test_classes, labels))
    confusion_mat = confusion_mat / np.sum(confusion_mat,axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_mat)
    fig.colorbar(cax)
    ax.set_xticks(range(len(test_classes)))
    ax.set_xticklabels(test_classes ,rotation=45, ha='left', rotation_mode='anchor')
    ax.set_yticks(range(len(train_classes)))
    ax.set_yticklabels(train_classes)
    plt.tight_layout()
    plt.xlabel('Test Classes', fontweight='bold')
    plt.ylabel('Train Classes', fontweight='bold')

    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig('figures/confusion_matrix.png', dpi=500)

def visualize_tsne_points(tx, ty, tx_c, ty_c, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Dict for All Classes
    # print("All Classes :", colors_per_class)
    # print("Train Classes : ", colors_per_class_train)
    # print("Test Classes : ", colors_per_class_test)

    # for every class, we'll add a scatter plot separately
    for train_label in colors_per_class_train:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == train_label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        
        current_tx_c = np.take(tx_c, indices)
        current_ty_c = np.take(ty_c, indices)
        
        """ print("<===============================================================\n")
        print("\nTrain Class Label:", train_label)
        print("\nTrain Class Indices:", indices)
        #print("current_tx:", current_tx)
        #print("current_ty:", current_ty)
        #print("current_tx_c:", current_ty_c)
        #print("current_ty_c:", current_ty_c)
        print("===============================================================>\n") """

        
        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class_train[train_label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, marker = "H", label=None)
        ax.scatter(current_tx_c, current_ty_c, c=color, marker = "p", label=train_label)
        #ax.scatter(centroid[:, 0], centroid[:, 1], marker= "+", c='black', s=200, alpha=0.5)
    # end train class plot

    # To plot the data points of the test classes.
    # Access all the test labels in the color_per_class_test dict()
    for test_label in colors_per_class_test:
        # find the samples of the current test class in the data
        indices = [i for i, l in enumerate(labels) if l == test_label]
        
        """ print("<===============================================================\n")
        print("\nTest Class Label:", test_label)
        print("\nTest Class Indices:", indices)
        print("===============================================================>\n") """
        
        current_tx = np.take(tx, indices[0])
        current_ty = np.take(ty, indices[0])
        
        current_tx_c = np.take(tx_c, indices[0])
        current_ty_c = np.take(ty_c, indices[0])

        #print("current_tx:", current_tx)
        #print("current_ty:", current_ty)
        #print("current_tx_c:", current_ty_c)
        #print("current_ty_c:", current_ty_c)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class_test[test_label][::-1]], dtype=np.float) / 255
    
        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, marker = "H", label=None)
        ax.scatter(current_tx_c, current_ty_c, c=color, marker = "p", label=test_label)
        #ax.scatter(centroid[:, 0], centroid[:, 1], marker= "+", c='black', s=200, alpha=0.5)

    # end test class plot
    
    # Call the function to calculate the distance between data points of train and test classes.
    euclidean_distance(tx_c, ty_c, labels)

    # Add multiple legends
    
    """ # Add first legend:  only labeled data is included
    train_legend = ax.legend(loc='lower left', title="Train Classes")
    # Add second legend for the test classes
    # train_legend will be removed from figure
    test_legend = ax.legend([test_label], loc='upper right', title="Test Classes")
    # Manually add the first legend back
    ax.add_artist(train_legend) """

    # Plot Title
    plt.title('Clustering results using AWA 2 Data Set')
    
    # build a legend using the labels we set previously
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot Legend
    ax.legend(loc='best')

    # Display the TSNE plot
    plt.show()

def visualize_tsne(tsne, tsne_c, images, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # To extract the x & y coordinates of the centers
    tx_c = tsne_c[:, 0]	
    ty_c = tsne_c[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # scale and move the coordinates so they fit [0; 1] range
    tx_c = scale_to_01_range(tx_c)
    ty_c = scale_to_01_range(ty_c)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, tx_c, ty_c, labels)

    # visualize the plot: samples as images
    #visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)

def main():
    
    parser = argparse.ArgumentParser()

    #parser.add_argument('--path', type=str, default='data/raw-img')
    #parser.add_argument('--path', type=str, default='data/JPEGImages_test')
    parser.add_argument('--path', type=str, default=PATH+'data/Animals_with_Attributes2/JPEGImages')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    args = parser.parse_args()

    fix_random_seeds()
    predicates = np.array(np.genfromtxt(PATH+'data/Animals_with_Attributes2/predicates.txt', dtype='str'))[:,-1]
    num_features = len(predicates)
    model = train_network(num_features,args.batch)

    features, labels, image_paths, labels_features_dict = get_features(
        batch=args.batch,
        num_images=args.num_images,
        model=model
    )
    
    # Change the value of centroid according to 50, 80, 100 ...
    labels, centroid = k_means_cluster(labels_features_dict,50)
    tsne = dict()
    tsne_c = dict() # plot classes
    tsne_pts = TSNE(n_components=2).fit_transform(features)
    tsne_centroid = TSNE(n_components=2).fit_transform(centroid)
    tsne_c['pts'] = tsne_pts
    tsne['centroid'] = tsne_centroid
    
    # Call the Function to plot the data points and images
    visualize_tsne(tsne['centroid'], tsne_c['pts'], image_paths, labels)

if __name__ == '__main__':
    main()
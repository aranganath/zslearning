import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import cv2
import random

#import pdb; pdb.set_trace()
# colors_per_class = {'antelope': [68, 233, 122], 'grizzly+bear': [225, 192, 22], 'killer+whale': [2, 120, 68], 'beaver': [99, 155, 187], 'dalmatian': [122, 160, 230], 'horse': [223, 240, 33], 'german+shepherd': [166, 80, 114], 'blue+whale': [211, 122, 18], 'siamese+cat': [16, 254, 154], 'chimpanzee': [36, 41, 76], 'giant+panda': [196, 191, 76], 'leopard': [57, 49, 226], 'persian+cat': [85, 97, 178], 'pig': [221, 212, 228], 'hippopotamus': [125, 140, 73], 'humpback+whale': [91, 61, 136], 'raccoon': [233, 154, 84], 'rat': [89, 90, 245], 'seal': [177, 167, 222]}

colors_per_class = dict()
colors_per_class_train = dict()
colors_per_class_test = dict()

class AnimalDataset(data.dataset.Dataset):
  def __init__(self, classes_file, transform,PATH):
    predicate_binary_mat = np.array(np.genfromtxt(PATH+'data/Animals_with_Attributes2/predicate-matrix-binary.txt', dtype='int'))
    self.predicate_binary_mat = predicate_binary_mat
    self.transform = transform
    
    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(PATH+'data/Animals_with_Attributes2/classes.txt') as f: 
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip() # reading data from classes.txt file
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    img_names = []
    img_index = []
    img_paths = []
    
    # To Read the classes data from the Actual data path
    with open(PATH+'data/Animals_with_Attributes2/{}'.format(classes_file)) as f:
      for line in f:
        class_name = line.strip()
        FOLDER_DIR = os.path.join(PATH+'data/Animals_with_Attributes2/JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor) # multiple files
        
        # Adjust the value of number of classes by changing the class index values.
        if(class_to_index[class_name] ==50):
                break
        
        # Generate the colors value at random
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(0,255)

        rgb_color = [red, green, blue]
        # Add the all the train & test classes to the colors_per_class dict()
        colors_per_class[class_name] = rgb_color

        # Add the train classes to the colors_per_class_train dict()
        if(classes_file == 'trainclasses.txt'):
          colors_per_class_train[class_name] = rgb_color
        
        # Add the test classes to the colors_per_class_test dict()
        if(classes_file == 'testclasses.txt'):
          colors_per_class_test[class_name] = rgb_color

        class_index = class_to_index[class_name] #class indexing
           
        for file_name in files:
          img_paths.append(file_name)
          img_names.append(class_name)
          img_index.append(class_index) # class index value will be stored in img_index[]
    self.img_paths = img_paths
    self.img_names = img_names
    self.img_index = img_index
    #print(colors_per_class) 
  
  def __getitem__(self, index):
    im = Image.open(self.img_paths[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)
    if im.shape != (3,224,224):
      print(self.img_names[index])

    im_index = self.img_index[index]
    im_predicate = self.predicate_binary_mat[im_index,:] #till last index value
    return im, im_predicate, self.img_names[index], im_index

  def __len__(self):
    return len(self.img_names)
'''
  def generate_class_list(self):
  # can be generated randomly
      self.custom_class_names = []
      
      # read class name from i to 10
      with open('tmp/data/Animals_with_Attributes2/classes.txt') as f: 
          index = 0
          for line in f:
              class_name = line.split('\t')[1].strip() # reading data from classes.txt file
              #custom_class_names[index] = class_name
              self.custom_class_names.append(class_name)
              index += 1
              #if(index==10):
              #    break 
      return self.custom_class_names

  def generate_colors_per_class(self):
      colors_per_class = {}

      #red = random.randint(0,255)
      #green = random.randint(0,255)
      #blue = random.randint(0,255)

      #rgb_color = [red, green, blue]

      for i in range(10):
          red = random.randint(0,255)
          green = random.randint(0,255)
          blue = random.randint(0,255)

          rgb_color = [red, green, blue]
          
          self.custom_class_names = self.generate_class_list()

          colors_per_class[self.custom_class_names[i]] = rgb_color
          print(colors_per_class)
      
      #return colors_per_class
  '''  
if __name__ == '__main__':
  dataset = AnimalDataset('testclasses.txt')




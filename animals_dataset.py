from os import path, listdir
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# processes Animals10 dataset: https://www.kaggle.com/alessiocorrado99/animals10
class AnimalsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_images=1000):
            
            custom_class_names = generate_class_list()
                        
            #print to debug (can be removed later)
            print("The following classes are used in training:")
            for i in range(10):
                print(custom_class_names[i])
                
            if not path.exists(data_path):
                raise Exception(data_path + ' does not exist!')

            self.data = []

            folders = listdir(data_path)
            for i in range(10):
                    label = custom_class_names[i]
                    full_path = path.join(data_path, custom_class_names[i])
                    images = listdir(full_path)

                    current_data = [(path.join(full_path, image), label) for image in images]
                    self.data += current_data

            num_images = min(num_images, len(self.data))
            self.data = random.sample(self.data, num_images) # only use num_images images

            # We use the transforms described in official PyTorch ResNet inference example:
            # https://pytorch.org/hub/pytorch_vision_resnet/.
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path, label = self.data[index]

        image = Image.open(image_path)

        try:
            image = self.transform(image) # some images in the dataset cannot be processed - we'll skip them
        except Exception:
            return None

        dict_data = {
            'image' : image,
            'label' : label,
            'image_path' : image_path
        }
        return dict_data


# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)

def generate_class_list():
    # can be generated randomly
    custom_class_names = []

    # read class name from i to 10
    with open('data/classes.txt') as f: 
        index = 0
        for line in f:
            class_name = line.split('\t')[1].strip() # reading data from classes.txt file
            #custom_class_names[index] = class_name
            custom_class_names.append(class_name)
            index += 1
            if(index==10):
                break
    return custom_class_names

def generate_colors_per_class():
    colors_per_class = {}

    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(0,255)

    rgb_color = [red, green, blue]

    for i in range(10):
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(0,255)

        rgb_color = [red, green, blue]
        
        custom_class_names = generate_class_list()

        colors_per_class[custom_class_names[i]] = rgb_color
        #print(colors_per_class)
    
    return colors_per_class


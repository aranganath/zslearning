# Author - Puneet
# Downloading dataset, creating directories, extracting and deleting
import requests
from zipfile import ZipFile 
import os

temp_Directory = "/tmp/data/"
parent_Directory = os.getcwd()
dataset_url = 'https://cvml.ist.ac.at/AwA2/AwA2-data.zip'
file_name = 'AwA2-data.zip'

#This API is used to create a temporary directory in the /tmp/ folder in linux
#This way the dataset is automatically deleted upon restart
def createDirectory():

    file_name = 'AwA2-data.zip'
    check_data_path = temp_Directory+file_name
    #Check if the dataset is available in the given path above
    if os.path.isfile(check_data_path):
        print("Dataset " + check_data_path + " already exists!\n")
        return True
    #If not, create the folder
    else:
        print("Dataset " +check_data_path+ " does not exist. Checking if folder exists...\n")
        #check if the directory has been created. Ignore if it has already been created
        if(os.path.isdir(temp_Directory)):
            print("Path already exists. No need to create directory\n")
            return False
        else:
        #Create the directory
            print("Path" + temp_Directory + "does not exist. Creating now... \n")
            try:
                os.makedirs(temp_Directory)
            except OSError:
                print ("Creation of the directory %s failed : \n" % temp_Directory)
            else:
                print ("Successfully created the directory %s: \n" % temp_Directory)
        return False
    
#This API is use to download the dataset
#IF the dataset already exists, there is no need to actually download the dataset
def DloadDataset():
    data_file = temp_Directory+file_name
    print("Downloading dataset from source. Path: "+data_file+"\n\n")
    r = requests.get(dataset_url, allow_redirects = True)
    #Downloading the file to the path created in createDirectory()
    
    with open(data_file, 'wb') as file: 
        file.write(r.content)

    print("Dataset files downloaded from source.. \n\n")
       
def unzipDataset():
    zip_dir = temp_Directory+file_name
    with ZipFile(zip_dir, 'r') as zip:
        for member in zip.namelist():
            checkdir = temp_Directory+member
            if os.path.exists(checkdir) or os.path.isfile(checkdir):
                print ("Error ",member," exists")
            else:
                zip.extract(member, path=temp_Directory)
                print("Extracting", member,"here")

def deleteDirectory():
    deleteDirectory = "/temp_test"
    delete_Directory_Path = os.getcwd() + deleteDirectory
    deletePath = delete_Directory_Path
    try:
        os.remove(deleteDirectory)
    except OSError:
        print ("Deletion of the directory %s failed :\n" % deleteDirectory)
    else:
        print ("Successfully deleted the directory %s : \n" % deleteDirectory)
        
def main(): 
    '''temp_Directory = "/temp_test/data/"
    parent_Directory = os.getcwd()
 
    dataset_url = 'https://cvml.ist.ac.at/AwA2/AwA2-base.zip'
    file_name = dataset_url.split('/')[-1]
    '''
    print("Downloading and Preparing Dataset for Zero Shot Leaning .. ..\n\n")
   
    if not createDirectory():
        DloadDataset()
    
    unzipDataset()
    #deleteDirectory()

if __name__ == "__main__": 
    main() 
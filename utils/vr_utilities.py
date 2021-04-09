
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- PATH ---

def get_file_name(file_path):
    return os.path.basename(file_path)

def get_file_paths(folder_path,search_pattern,search_in_subfolders=False):
    if (search_in_subfolders):
        pathname=folder_path+'/**/'+search_pattern
    else:
        pathname=folder_path+'/'+search_pattern

    file_paths=glob.glob(pathname,recursive=search_in_subfolders)

    return [path.replace('\\','/') for path in file_paths]

# ---

# --- DATASET ---

def load_labeled_dataset(filelist_name, db_folder_path):
    files = []
    labels = []
    
    db_folder_path = Path(db_folder_path)
    
    with open(str(db_folder_path / filelist_name), 'r') as f:
        for i, line in enumerate(f):
            path, label = line.split()
            files.append(db_folder_path / path)
            labels.append(int(label))

    images = [plt.imread(file_path) for file_path in files]

    return np.array(images), np.array(labels)
	
def load_label_names(filelabel_name, db_folder_path):
    with open(str(Path(db_folder_path) / filelabel_name)) as f:
        content = f.readlines()
            
    return np.array([x.strip() for x in content])

def load_image_dataset_with_masks(image_folder_path,mask_folder_path,image_search_pattern,image_count=None):
    image_file_path_list=get_file_paths(image_folder_path,image_search_pattern)

    images=[]
    masks=[]
    for image_file_path in image_file_path_list:
        if (image_count!=None and len(images)==image_count):
          break
        
        image=plt.imread(image_file_path)
        mask=plt.imread(mask_folder_path+'/'+get_file_name(image_file_path))

        if (len(image.shape)==2):
          image=image[:,:,np.newaxis]
        
        mask=mask[:,:,np.newaxis]

        images.append(image)
        masks.append(mask)

    return np.array(images),np.array(masks)
	
# ---
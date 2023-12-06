"""
Title: Custom Datasets
Description: Custom Datasets for 3D Point Clouds
"""

import os
import numpy as np
import warnings
import pickle
import h5py

from torch.utils.data import Dataset

# Normalize point cloud?
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# Load hdf5 file
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:] # .astype(np.int32)
    return (data, label)
    

class ModelNetDataLoader(Dataset):
    """ Dataset loader for https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip """
    def __init__(self, root, 
                 num_point,
                 use_uniform_sample,
                 use_normals,
                 num_category, 
                 split='train', 
                 process_data=False):
        self.root           = root                      # root path
        self.npoints        = num_point            # number of points
        
        self.process_data   = process_data
        self.uniform        = use_uniform_sample
        self.use_normals    = use_normals
        self.num_category   = num_category         # ModelNet10 vs ModelNet40
        
        if self.num_category == 10:
            self.category_file = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.category_file = os.path.join(self.root, 'modelnet40_shape_names.txt')
            
        self.categories = [line.rstrip() for line in open(self.category_file)]
        self.classes = dict(zip(self.categories, range(len(self.categories))))
        self.idx_classes = dict(zip(range(len(self.categories)), self.categories))
        
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
            
        # Set the path of the files
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        
        # TODO: Process data
        if self.process_data:
            pass

    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        # TODO: Process data
        
        if self.process_data:
            pass
        else: 
            fn      = self.datapath[index]                    # filename
            cls     = self.classes[self.datapath[index][0]]
            label   = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
            # TODO: Uniform Points
            if self.uniform:
                pass
            else:
                point_set = point_set[0:self.npoints, :]      # Take only the first npoints
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3] # First 3 Points

        return point_set, label[0]
        
class ModelNetPlyHDF52048DataLoader(Dataset):
    """ Dataloader for https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip 
    Training: 9840
    Testing: 4096
    Density: 2048 points per point cloud
    
    # TODO - Implement processing of data (normalization, etc.) For now, just load the data as is
    """
    def __init__(self, root, num_point, split="train", process_data = True):
        self.root = root
        self.npoints = num_point
        self.process_data = process_data
        self.split = split.lower()
        
        file_list_name = 'train_files.txt' if self.split == "train" else 'test_files.txt'
        self.files = [line.rstrip() for line in open(os.path.join(self.root, file_list_name))]
        
        # Preload data
        self.point_sets = []
        self.labels = []
        for f in self.files:
            point_set, label = load_h5(f)
            self.point_sets.append(point_set)
            self.labels.append(label)

        # Get category information
        self.categories = [line.rstrip() for line in open(os.path.join(self.root, 'shape_names.txt'))]
        self.idx_classes = dict(zip(range(len(self.categories)), self.categories))
        self.classes_idx = dict(zip(self.categories, range(len(self.categories))))
        
    
    def __len__(self):
        # TODO Hardcoded for now
        if self.split == "train":
            return 9840  # 2048 + 2048 + 2048 + 2048 + 1468
        elif self.split == "test":
            return 2048 + 420
    
    def __getitem__(self, idx):
        file_idx = idx // 2048  # index div 2048 to get the file index
        point_idx = idx % 2048  # index mod 2048 to get the point index
        
        # # Load the file
        # point_set, label = load_h5(self.files[file_idx])
        
        # # Get point_idx in point_set
        # point_set = point_set[point_idx]
        # point_set = point_set[:self.npoints, :]
        
        point_set = self.point_sets[file_idx][point_idx][:self.npoints, :]
        label = self.labels[file_idx][point_idx].squeeze()
        
        return point_set, label
        

        # return point_set, label[point_idx].squeeze()
 
 
class ShapeNetCoreDataLoader(Dataset):
    """ Dataloader forhttps://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip
     
    """
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
        
if __name__ == "__main__":
    pass
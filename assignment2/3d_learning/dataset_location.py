# specify the root location where u downloaded the dataset
root_location = "/media/thaoanh/NewVolume/Minh" 
use_full_dataset = False
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

R2N2_PATH = f"/home/nhvtmjnh/projects/3d/assignment2/3d_learning/data/r2n2_shapenet_dataset/r2n2"
SHAPENET_PATH = f"/home/nhvtmjnh/projects/3d/assignment2/3d_learning/data/r2n2_shapenet_dataset/shapenet"

if use_full_dataset:
    SPLITS_PATH = f"/home/nhvtmjnh/projects/3d/assignment2/3d_learning/data/r2n2_shapenet_dataset/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"/home/nhvtmjnh/projects/3d/assignment2/3d_learning/data/r2n2_shapenet_dataset/split_03001627.json"  # split file contains data entry for 03001627 class

import torch
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
import kornia  # For quaternion to rotation matrix conversion

# Dataset class adjusted for your dataset structure
class VisualLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.gt_data = self._load_ground_truth()

        # Load samples
        self.samples = self._load_samples()

        print(f"Number of samples: {len(self.samples)}")

    def _load_ground_truth(self):
        gt_path = os.path.join(self.root_dir, 'GT.txt')
        gt_data = {}
        with open(gt_path, 'r') as f:
            lines = f.readlines()[2:]  # Skip the first two lines (header)
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue  # Skip malformed lines
                image_path = parts[0]  # Image path relative to the dataset root
                pose = np.array([float(x) for x in parts[1:]])
                gt_data[os.path.normpath(image_path)] = pose
        return gt_data

    def _load_samples(self):
        samples = []
        for seq_folder in os.listdir(self.root_dir):
            seq_path = os.path.join(self.root_dir, seq_folder)
            if os.path.isdir(seq_path):
                image_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                for image_file in image_files:
                    image_path = os.path.join(seq_folder, image_file)
                    norm_image_path = os.path.normpath(image_path)
                    if norm_image_path in self.gt_data:
                        full_image_path = os.path.join(self.root_dir, image_path)
                        samples.append((full_image_path, self.gt_data[norm_image_path]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        color_path, pose = self.samples[idx]
        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        translation = pose[:3]
        quaternion = pose[3:]
        pose_matrix = np.concatenate([translation, quaternion])
        if self.transform:
            color_image = self.transform(color_image)
        return color_image, torch.tensor(pose_matrix, dtype=torch.float32)

# Function to parse reconstruction.nvm and extract 3D world points (w_P)
def parse_nvm_file(nvm_file_path):
    scene_coordinates = []
    with open(nvm_file_path, 'r') as file:
        lines = file.readlines()
        n_views = int(lines[2])  # Number of images
        n_points = int(lines[2 + n_views + 2])  # Number of 3D points

        for i in range(3 + n_views + 3, 3 + n_views + 3 + n_points):
            point_data = lines[i].strip().split()[:3]  # Extract only the first 3 values (XYZ coordinates)

            # Ensure the line contains valid 3D coordinates (3 floating-point numbers)
            if len(point_data) != 3:
                print(f"Skipping invalid point on line {i}: {lines[i].strip()}")
                continue  # Skip this line if it doesn't have exactly 3 values

            try:
                w_P = np.array([float(coord) for coord in point_data])
                scene_coordinates.append(w_P)  # Append valid 3D coordinates to scene_coordinates
            except ValueError:
                print(f"Skipping malformed point data on line {i}: {lines[i].strip()}")
                continue  # Skip lines with invalid float conversions

    if len(scene_coordinates) == 0:
        raise ValueError("No valid 3D points found in the reconstruction.nvm file.")

    return torch.tensor(scene_coordinates, dtype=torch.float32)

# Function to calculate xmin and xmax for each image in the dataset
def compute_xmin_xmax(w_P, c_R_w, w_t_c, xmin_percentile=0.025, xmax_percentile=0.975):
    # Project world points to camera frame
    c_P = c_R_w @ (w_P.T - w_t_c)

    # Depth values from the Z-axis (third column of c_P)
    depths = c_P[2, :]

    # Filter depths based on valid range (as in original code)
    valid_depths = depths[(depths > 0.2) & (depths < 1000)]

    # Sort valid depths to compute percentiles
    sorted_depths = torch.sort(valid_depths).values

    # Compute xmin and xmax using specified percentiles
    xmin = sorted_depths[int(xmin_percentile * (sorted_depths.shape[0] - 1))]
    xmax = sorted_depths[int(xmax_percentile * (sorted_depths.shape[0] - 1))]

    return xmin, xmax

# Convert quaternion to rotation matrix using Kornia
def quaternion_to_rotation_matrix(quaternion):
    """
    Converts a quaternion to a 3x3 rotation matrix using Kornia.
    Kornia expects quaternions in (batch_size, 4) format.
    """
    # Ensure the quaternion is in the correct format: (4,) -> (1, 4)
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)  # Add a batch dimension

    # Ensure quaternion is normalized (optional, depending on your data)
    quaternion = kornia.geometry.quaternion.normalize_quaternion(quaternion)

    # Convert to rotation matrix
    rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternion)

    # Return the first (and only) rotation matrix if there was only one quaternion
    return rotation_matrix[0] if rotation_matrix.shape[0] == 1 else rotation_matrix

# Main block for computing xmin and xmax for the dataset
if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/KingsCollege"  # Update this with your dataset path
    nvm_file_path = os.path.join(dataset_path, 'reconstruction.nvm')

    # Load the dataset and parse the 3D world points (w_P)
    dataset = VisualLandmarkDataset(root_dir=dataset_path)
    w_P = parse_nvm_file(nvm_file_path)

    global_xmin = []
    global_xmax = []

    # Using tqdm for progress bar
    for i, (image, pose) in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Images"):
        translation = pose[:3].view(3, 1)
        quaternion = pose[3:]  # Quaternion
        c_R_w = quaternion_to_rotation_matrix(quaternion)  # Using Kornia for quaternion to rotation matrix conversion

        # Compute xmin and xmax for this image
        xmin, xmax = compute_xmin_xmax(w_P, c_R_w, translation)

        global_xmin.append(xmin)
        global_xmax.append(xmax)

    # Compute global xmin and xmax for the dataset
    global_xmin = torch.min(torch.tensor(global_xmin))
    global_xmax = torch.max(torch.tensor(global_xmax))

    print(f"Global xmin: {global_xmin.item()}")
    print(f"Global xmax: {global_xmax.item()}")

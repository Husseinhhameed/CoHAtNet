def load_and_process_poses(data_dir, seqs, train=True, real=False, vo_lib='orbslam'):
    """
    Loads and processes pose data from the given directory.

    Args:
        data_dir (str): The root directory containing the dataset.
        seqs (list): List of sequences to load poses from.
        train (bool): If True, compute and save statistics; otherwise, load them.
        real (bool): If True, use real-world poses; otherwise, use synthetic ones.
        vo_lib (str): Visual odometry library used ('orbslam', 'libviso2', etc.).

    Returns:
        np.ndarray: Processed and normalized poses.
    """
    ps = {}  # Dictionary to store pose data for each sequence
    vo_stats = {}  # Dictionary to store VO statistics for each sequence
    all_poses = []  # List to collect all pose data

    for seq in seqs:
        seq_dir = os.path.join(data_dir, seq)
        p_filenames = [n for n in os.listdir(seq_dir) if n.find('pose') >= 0]

        if real:
            # Load poses from a real-world dataset
            pose_file = os.path.join(data_dir, f'{vo_lib}_poses', seq)
            pss = np.loadtxt(pose_file)
            frame_idx = pss[:, 0].astype(int)
            if vo_lib == 'libviso2':
                frame_idx -= 1
            ps[seq] = pss[:, 1:13]
            vo_stats_filename = os.path.join(seq_dir, f'{vo_lib}_vo_stats.pkl')
            with open(vo_stats_filename, 'rb') as f:
                vo_stats[seq] = pickle.load(f)
        else:
            # Load poses from synthetic dataset
            frame_idx = np.array(range(len(p_filenames)), dtype=int)
            pss = [np.loadtxt(os.path.join(seq_dir, f'frame-{i:06d}.pose.txt')).flatten()[:12]
                   for i in frame_idx
                   if os.path.exists(os.path.join(seq_dir, f'frame-{i:06d}.pose.txt'))]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}  # Default VO stats for synthetic data

        all_poses.append(ps[seq])

    all_poses = np.vstack(all_poses)
    pose_stats_filename = os.path.join(data_dir, 'pose_stats.txt')

    # Compute or load statistics for normalization
    if train and not real:
        mean_t = np.mean(all_poses[:, [3, 7, 11]], axis=0)
        std_t = np.std(all_poses[:, [3, 7, 11]], axis=0)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
    else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

    # Process and normalize poses
    processed_poses = []
    for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                            align_s=vo_stats[seq]['s'])
        processed_poses.append(pss)

    return np.vstack(processed_poses)


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    Aligns and normalizes poses.

    Args:
        poses_in (np.ndarray): Input poses to process.
        mean_t (np.ndarray): Mean translation for normalization.
        std_t (np.ndarray): Standard deviation of translation for normalization.
        align_R (np.ndarray): Rotation matrix for alignment.
        align_t (np.ndarray): Translation vector for alignment.
        align_s (float): Scaling factor for alignment.

    Returns:
        np.ndarray: Processed poses.
    """
    poses_out = np.zeros((len(poses_in), 7))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]  # Translation components

    # Align and process rotation
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # Constrain to hemisphere
        poses_out[i, 3:] = q  # Keep the quaternion as 4D
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # Normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    return poses_out

# Define the transformation for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
import random

class FireDataset(Dataset):
    def __init__(self, root_dir, xmin_percentile=0.025, xmax_percentile=0.975, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.xmin_percentile = xmin_percentile
        self.xmax_percentile = xmax_percentile

        # Load sequences and samples
        self.seqs = [seq for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, seq))]
        self.samples = self._load_samples()

        # Check for precomputed depth statistics
        stats_file = os.path.join(root_dir, 'depth_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.global_xmin = stats['global_xmin']
                self.global_xmax = stats['global_xmax']
        else:
            # Compute depth statistics using a subset
            self.global_depths = self._compute_global_depths()
            self.global_xmin = np.percentile(self.global_depths, self.xmin_percentile * 100)
            self.global_xmax = np.percentile(self.global_depths, self.xmax_percentile * 100)
            with open(stats_file, 'w') as f:
                json.dump({'global_xmin': self.global_xmin, 'global_xmax': self.global_xmax}, f)
            del self.global_depths  # Free memory

        # Load and process poses
        self.processed_poses = self._load_processed_poses()

        # Ensure consistency between samples and processed poses
        min_length = min(len(self.samples), self.processed_poses.shape[0])
        self.samples = self.samples[:min_length]
        self.processed_poses = self.processed_poses[:min_length]

    def _load_samples(self):
        """
        Loads all sample file paths from the dataset directory.
        """
        samples = []
        for seq_folder in self.seqs:
            seq_path = os.path.join(self.root_dir, seq_folder)
            color_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.color.png')])
            depth_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.depth.png')])
            pose_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.pose.txt')])

            for color_file, depth_file, pose_file in zip(color_files, depth_files, pose_files):
                samples.append((os.path.join(seq_path, color_file),
                                os.path.join(seq_path, depth_file),
                                os.path.join(seq_path, pose_file)))
        return samples

    def _compute_global_depths(self):
        """
        Computes global depth statistics by loading a subset of depth images.
        """
        num_samples = len(self.samples)
        sample_indices = random.sample(range(num_samples), min(100, num_samples))  # Use a subset of images
        all_depths = []

        for idx in tqdm(sample_indices, desc="Computing global depths"):
            _, depth_path, _ = self.samples[idx]
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            valid_depths = depth_image[depth_image > 0]
            all_depths.extend(valid_depths.flatten())

        all_depths = np.array(all_depths)
        return all_depths

    def _load_processed_poses(self):
        """
        Loads and processes pose data from the given directory.
        """
        return load_and_process_poses(self.root_dir, self.seqs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.processed_poses):
            raise IndexError(f"Index {idx} out of bounds for processed poses of size {len(self.processed_poses)}")

        color_path, depth_path, _ = self.samples[idx]

        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pose_matrix = self.processed_poses[idx]

        if self.transform:
            color_image = self.transform(color_image)
            depth_image = (depth_image / depth_image.max() * 255).astype(np.uint8)
            depth_image = self.transform(depth_image)

        return color_image, depth_image, pose_matrix

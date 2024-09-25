# Dataset class adjusted for new dataset structure
class VisualLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.gt_data = self._load_ground_truth()
        self.samples = self._load_samples()

    def _load_ground_truth(self):
        gt_path = os.path.join(self.root_dir, 'GT.txt')
        gt_data = {}
        with open(gt_path, 'r') as f:
            lines = f.readlines()[2:]  # Skip the first two lines (header)
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                image_path = parts[0]
                pose = np.array([float(x) for x in parts[1:]])  # Includes [X Y Z W P Q R]
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
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for samples of size {len(self.samples)}")
        color_path, pose = self.samples[idx]
        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        translation = pose[:3]  # [X, Y, Z]
        quaternion = pose[3:]   # [W, P, Q, R] (quaternion)
        pose_matrix = np.concatenate([translation, quaternion])
        if self.transform:
            color_image = self.transform(color_image)
        return color_image, torch.tensor(pose_matrix, dtype=torch.float32)

# Define the transformation for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

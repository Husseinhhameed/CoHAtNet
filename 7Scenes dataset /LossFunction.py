# Define the Global Homography Loss Function
class GlobalHomographyLoss(torch.nn.Module):
    def __init__(self, xmin, xmax, device='cpu'):
        """
        `xmin` is the minimum distance of observations across all frames.
        `xmax` is the maximum distance of observations across all frames.
        """
        super().__init__()

        # `xmin` is the minimum distance of observations in all frames
        xmin = torch.tensor(xmin, dtype=torch.float32, device=device)

        # `xmax` is the maximum distance of observations in all frames
        xmax = torch.tensor(xmax, dtype=torch.float32, device=device)

        # `B_weight` and `C_weight` are the weights of matrices A and B computed from `xmin` and `xmax`
        self.B_weight = torch.log(xmin / xmax) / (xmax - xmin)
        self.C_weight = xmin * xmax

        # `c_n` is the normal vector of the plane inducing the homographies in the ground-truth camera frame
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)

        # `eye` is the (3, 3) identity matrix
        self.eye = torch.eye(3, device=device)

    def forward(self, batch):
        A, B, C = compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)

        error = A + B * self.B_weight + C / self.C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error

  def compute_ABC(w_t_c, c_R_w, w_t_chat, chat_R_w, c_n, eye):
    """
    Computes A, B, and C matrix given estimated and ground truth poses
    and normal vector n.
    `w_t_c` and `w_t_chat` must have shape (batch_size, 3, 1).
    `c_R_w` and `chat_R_w` must have shape (batch_size, 3, 3).
    `n` must have shape (3, 1).
    `eye` is the (3, 3) identity matrix on the proper device.
    """
    # Ensure all inputs are float32
    w_t_c = w_t_c.float()
    c_R_w = c_R_w.float()
    w_t_chat = w_t_chat.float()
    chat_R_w = chat_R_w.float()
    c_n = c_n.float()
    eye = eye.float()

    chat_t_c = chat_R_w @ (w_t_c - w_t_chat)
    chat_R_c = chat_R_w @ c_R_w.transpose(1, 2)

    A = eye - chat_R_c
    C = c_n @ chat_t_c.transpose(1, 2)
    B = C @ A
    A = A @ A.transpose(1, 2)
    B = B + B.transpose(1, 2)
    C = C @ C.transpose(1, 2)

    return A, B, C
def convert_to_transformation_matrix(translation, quaternion):
    """
    Converts a translation vector and quaternion to a 4x4 transformation matrix.
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)

    # Create a 4x4 transformation matrix
    T = torch.eye(4)  # Initialize as 4x4 identity matrix
    T[:3, :3] = R  # Set the top-left 3x3 submatrix as the rotation matrix
    T[:3, 3] = translation  # Set the top-right 3x1 subvector as the translation vector

    return T

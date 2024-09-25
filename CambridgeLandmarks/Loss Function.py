class GlobalHomographyLoss(torch.nn.Module):
    def __init__(self, xmin, xmax, device='cpu'):
        super().__init__()
        xmin = torch.tensor(xmin, dtype=torch.float32, device=device)
        xmax = torch.tensor(xmax, dtype=torch.float32, device=device)
        self.B_weight = torch.log(xmin / xmax) / (xmax - xmin)
        self.C_weight = xmin * xmax
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)
        self.eye = torch.eye(3, device=device)

    def forward(self, batch):
        A, B, C = compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)
        error = A + B * self.B_weight + C / self.C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error

# Compute A, B, and C matrices for homography-based error
def compute_ABC(w_t_c, c_R_w, w_t_chat, chat_R_w, c_n, eye):
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


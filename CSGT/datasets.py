from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ItemDataset(Dataset):
    def __init__(self, text_emb_np, cf_emb_np=None, sigma_np=None, knn_k=20, mode="train"):
        """
        text_emb_np: np.ndarray or path to a .npy file (shape: N x D_text)
        cf_emb_np:  np.ndarray or path to a .npy file (shape: N x D_cf). Required in training mode; can be None in test mode.
        sigma_np:   1D array of shape (N,) or None. If None, defaults to all ones.
        knn_k:      Number of CF neighbors to precompute during training.
        mode:       "train" or "test"
        """
        if isinstance(text_emb_np, str):
            text_emb_np = np.load(text_emb_np)  # (N, D_text)
        self.text_emb = torch.from_numpy(text_emb_np).float()
        self.N = self.text_emb.shape[0]
        self.text_dim = self.text_emb.shape[1]
        self.knn_k = knn_k
        self.mode = mode

        if mode == "train":
            if isinstance(cf_emb_np, str):
                cf_emb_np = np.load(cf_emb_np)  # (N, D_cf)
            assert cf_emb_np is not None, "cf_emb_np is required in train mode"

            self.cf_emb = torch.from_numpy(cf_emb_np).float()
            self.cf_dim = self.cf_emb.shape[1]

            if sigma_np is None:
                sigma_np = np.ones(self.N, dtype=np.float32)
            self.sigma = torch.from_numpy(sigma_np).float()

            # 预计算全局 CF kNN
            nbrs = NearestNeighbors(n_neighbors=min(knn_k+1, self.N), metric='cosine').fit(cf_emb_np)
            _, indices = nbrs.kneighbors(cf_emb_np)
            self.global_knn = indices[:, 1:min(knn_k+1, self.N)]

        else:  # test mode
            self.cf_emb = None
            self.cf_dim = None
            self.sigma = None
            self.global_knn = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = {
            "id": int(idx),
            "text_emb": self.text_emb[idx]
        }
        if self.mode == "train":
            sample.update({
                "cf_emb": self.cf_emb[idx],
                "sigma": self.sigma[idx],
                "global_knn": torch.tensor(self.global_knn[idx], dtype=torch.long)
            })
        return sample

class EmbDataset(Dataset):
    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)
import json
import os
import numpy as np
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# -------- 用户可选配置 --------
data_dir = "../data/Arts"
MODEL_NAME = "intfloat/e5-large-v2"   # "intfloat/e5-large-v2" | "BAAI/bge-large-en"
ITEM_FILE = data_dir+"/Arts.item.json"
INTER_FILE = data_dir+"/Arts.inter.json"
ITEM_OUT = data_dir+"/Arts.emb-llama2_7b-td.npy"
POOLING = "time_decay"   # "mean" | "max" | "attention" | "time_decay"
USER_OUT = "Arts_user_"+POOLING+"_representation.npy"
DECAY_ALPHA = 0.1        # 时间衰减系数

# -------- 加载或计算物品 embedding --------
if os.path.exists(ITEM_OUT):
    print(f"Found {ITEM_OUT}, loading item embeddings...")
    item_embeddings = np.load(ITEM_OUT)
else:
    print(f"Computing item embeddings with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    with open(ITEM_FILE, "r", encoding="utf-8") as f:
        item_data = json.load(f)

    item_texts = []
    item_ids = sorted(item_data.keys(), key=lambda x: int(x))
    for iid in item_ids:
        title = item_data[iid].get("title", "")
        desc = item_data[iid].get("description", "")
        text = title + ". " + desc
        # 添加 prompt
        if "e5" in MODEL_NAME:
            text = "passage: " + text
        elif "bge" in MODEL_NAME:
            text = "represent: " + text
        item_texts.append(text)

    print(f"Encoding {len(item_texts)} items...")
    item_embeddings = model.encode(
        item_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(ITEM_OUT, item_embeddings)
    print(f"Item embeddings saved to {ITEM_OUT}, shape={item_embeddings.shape}")

# -------- 加载用户交互 --------
with open(INTER_FILE, "r", encoding="utf-8") as f:
    inter_data = json.load(f)

user_embeddings = []
user_ids = sorted(inter_data.keys(), key=lambda x: int(x))
hidden_dim = item_embeddings.shape[1]

print(f"Building {len(user_ids)} user representations with {POOLING} pooling...")

def aggregate_embeddings(emb_list, method="mean", alpha=0.1):
    emb_tensor = torch.tensor(emb_list, dtype=torch.float32)

    if method == "mean":
        return emb_tensor.mean(dim=0).numpy()

    elif method == "max":
        return emb_tensor.max(dim=0).values.numpy()

    elif method == "attention":
        query = emb_tensor.mean(dim=0, keepdim=True)  # [1, d]
        attn_scores = torch.matmul(emb_tensor, query.T).squeeze(-1)  # [n]
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(1)    # [n,1]
        weighted_sum = (emb_tensor * attn_weights).sum(dim=0)
        return weighted_sum.numpy()

    elif method == "time_decay":
        n = emb_tensor.shape[0]
        weights = torch.tensor([np.exp(-alpha * (n - 1 - k)) for k in range(n)], dtype=torch.float32)
        weights = weights / weights.sum()
        weighted_sum = (emb_tensor * weights.unsqueeze(1)).sum(dim=0)
        return weighted_sum.numpy()

    else:
        raise ValueError(f"Unknown pooling method: {method}")

for uid in tqdm(user_ids):
    item_list = inter_data[uid][:-2]      # train phase history
    if len(item_list) == 0:
        user_embeddings.append(np.zeros(hidden_dim))
    else:
        emb = item_embeddings[item_list]
        user_embeddings.append(aggregate_embeddings(emb, method=POOLING, alpha=DECAY_ALPHA))

user_embeddings = np.vstack(user_embeddings)
np.save(USER_OUT, user_embeddings)
print(f"User embeddings saved to {USER_OUT}, shape={user_embeddings.shape}")

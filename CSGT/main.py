import argparse
import random
import torch
import numpy as np
from time import time
import logging
from torch.utils.data import DataLoader

from datasets import EmbDataset,ItemDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=100, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="../data", help="Input data path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument('--use_constrain_loss', type=bool, default=False, help='use constrain loss')
    parser.add_argument('--use_cf_loss', type=bool, default=False, help='use cf loss')
    parser.add_argument('--use_kd', type=bool, default=False, help='use knowledge distillation')
    parser.add_argument('--use_knn', type=bool, default=False, help='use knn loss')
    parser.add_argument('--knn_k', type=int, default=20, help='knn similar user number')
    parser.add_argument('--knn_weight', type=float, default=0.5, help='knn loss weight')
    parser.add_argument('--use_knn_argument', type=bool, default=False, help='use knn contrastive loss with argument')
    parser.add_argument('--knn_argument_alpha', type=float, default=0.5, help='knn contrastive loss argument alpha')
    parser.add_argument('--use_pairwise', type=bool, default=False, help='use pairwise loss')
    parser.add_argument('--pair_weight', type=float, default=0.2, help='pairwise loss weight')
    parser.add_argument('--pair_margin', type=float, default=0.1, help='pairwise loss margin')

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb', type=str, default="./RQ-VAE/ckpt/Instruments-32d-sasrec.pt", help='cf emb')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    print(args)
    logging.basicConfig(level=logging.DEBUG)
    cf_emb = np.load(args.cf_emb)

    """build dataset"""
    data = ItemDataset(args.data_path,args.cf_emb,knn_k=args.knn_k)#############参数
    print("text dim",data.text_dim)
    print("cf dim",data.cf_dim)
    model = RQVAE(in_dim=data.text_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  use_constrain_loss = args.use_constrain_loss,
                  use_cf_loss=args.use_cf_loss,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  cf_embedding = cf_emb,
                  use_kd=args.use_kd,
                  use_knn=args.use_knn,
                  knn_weight=args.knn_weight,
                  use_knn_argument=args.use_knn_argument,
                  knn_argument_alpha=args.knn_argument_alpha,
                  use_pairwise=args.use_pairwise,
                  pair_weight=args.pair_weight,
                  pair_margin=args.pair_margin,
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)
    import sys, torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit(0)





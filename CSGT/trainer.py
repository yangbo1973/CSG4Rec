import logging
import json
import numpy as np
import torch
import random
from time import time, strftime
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import EmbDataset,ItemDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()     # 同时打控制台
            ]
        )
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}


    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        # x = data.cpu().detach().numpy()
        # data = self.embedding.weight.cpu().detach().numpy()
        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data)    

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        # print(f'train_data len: {len(train_data)}')
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        
        if self.args.use_constrain_loss :
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            batch_items = []
            for i in range(len(data['text_emb'])):
                batch_items.append({
                    "id": int(data['id'][i].item()),
                    "text_emb": data['text_emb'][i].to(self.device),
                    "cf_emb": data['cf_emb'][i].to(self.device),
                    "sigma": data['sigma'][i].to(self.device),
                    "global_knn": data['global_knn'][i]  # keep on cpu for lookup
                })
            self.optimizer.zero_grad()

            t_loss = self.model.compute_loss(batch_items, epoch_idx ,self.epochs,self.labels)
            self._check_nan(t_loss['total'])
            t_loss['total'].backward()
            t_loss['total'] = t_loss['total'].item()
            self.optimizer.step()
        return t_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):
            batch_items = []
            for i in range(len(data['text_emb'])):
                batch_items.append({
                    "id": int(data['id'][i].item()),
                    "text_emb": data['text_emb'][i].to(self.device),
                    "cf_emb": data['cf_emb'][i].to(self.device),
                    "sigma": data['sigma'][i].to(self.device),
                    "global_knn": data['global_knn'][i]  # keep on cpu for lookup
                })
            text_emb = torch.stack([bi['text_emb'] for bi in batch_items])  # B x D_text
            num_sample += len(text_emb)
            text_emb = text_emb.to(self.device)
            indices = self.model.get_indices(text_emb, self.labels)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set))/num_sample
        # balance_score = self.balance_overall(tokens_appearance)
        # wandb.log({"collision_rate": collision_rate, "balance_score": 0})


        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, tloss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % tloss['total']
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % tloss['recon']
        train_loss_output +=", "
        train_loss_output += set_color("rq loss", "blue") + ": %.4f" % tloss['rq']
        train_loss_output +=", "
        train_loss_output += set_color("knn loss", "blue") + ": %.4f" % tloss['knn']
        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0
        # self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            t_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time,t_loss
            )
            self.logger.info(train_loss_output)
            if t_loss['total'] < self.best_loss:
                self.best_loss = t_loss['total']
                self._save_checkpoint(epoch=epoch_idx,ckpt_file=str(self.best_loss)+'_' +self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file= str(self.best_collision_rate)+'_' +self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                # if cur_eval_step >= 10:
                #     print("Finish!")
                #     break

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate





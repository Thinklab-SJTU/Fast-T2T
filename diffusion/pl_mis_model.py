import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_sparse import SparseTensor
from diffusion.co_datasets.mis_dataset import MISDataset

from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.pl_meta_model import COMetaModel
from diffusion.utils.mis_utils import mis_decode_np
from diffusion.consistency import MISConsistency


class MISModel(COMetaModel):
    def __init__(self, param_args=None, target_model=None, teacher_model=None, load_dataset=True):
        super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

        if load_dataset:
            # data_label_dir = None
            # if self.args.training_split_label_dir is not None:
            #     training_data_label_dir = os.path.join(self.args.training_split_label_dir)

            if self.args.training_split:
                self.train_dataset = MISDataset(
                    data_file=os.path.join(self.args.training_split),
                    data_label_dir=self.args.training_split_label_dir,
                )
            if self.args.test_split:
                self.test_dataset = MISDataset(
                    data_file=os.path.join(self.args.test_split),
                    data_label_dir=self.args.test_split_label_dir,
                )
            if self.args.validation_split:
                self.validation_dataset = MISDataset(
                    data_file=os.path.join(self.args.validation_split),
                    data_label_dir=self.args.validation_split_label_dir,
                )

        if self.args.consistency:
            self.consistency_trainer = MISConsistency(self.args, sigma_max=self.diffusion.T, boundary_func=self.args.boundary_func)

    def forward(self, x, t, edge_index):
        return self.model(x, t, edge_index=edge_index)

    def consistency_training_step(self, batch, batch_idx):
        loss = self.consistency_trainer.consistency_losses(self, batch)
        self.log("train/loss", loss)
        return loss

    def categorical_training_step(self, batch, batch_idx):
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        # Sample from diffusion
        node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()
        node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

        # print(t.shape)
        # print(node_labels_onehot.shape)
        xt = self.diffusion.sample(node_labels_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)  # N
        xt = xt.reshape(-1)  # N
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)  # 2, E

        # Denoise
        x0_pred = self.forward(
            xt.float().to(node_labels.device),
            t.float().to(node_labels.device),
            edge_index,
        )  # N, 2

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, node_labels)
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        if self.args.consistency:
            return self.consistency_training_step(batch, batch_idx)
        else:
            return self.categorical_training_step(batch, batch_idx)

    def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
        """

        Args:
            xt: B*N
            t: B*N OR B
            device:
            edge_index: E, 2
            target_t: B*N OR B

        Returns:

        """
        xt_scale = xt * 2 - 1
        xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))
        with torch.no_grad():
            x0_pred = self.forward(
                xt_scale,
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            xt, _ = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt

    def guided_categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
        torch.set_grad_enabled(True)
        xt = xt.float()  # n if sparse
        xt.requires_grad = True
        xt_scale = xt * 2 - 1
        xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))
        with torch.inference_mode(False):
            x0_pred = self.forward(
                xt_scale,
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            num_nodes = xt.shape[0]
            adj_matrix = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(num_nodes, num_nodes),
            ).to_dense()
            adj_matrix.fill_diagonal_(0)

            pred_nodes = x0_pred_prob[..., 1].squeeze(0)
            # cost_est = 1 - pred_nodes / num_nodes
            f_mis = -pred_nodes.sum()
            g_mis = adj_matrix @ pred_nodes
            g_mis = (pred_nodes * g_mis).sum()
            cost_est = f_mis + 0.5 * g_mis
            cost_est.requires_grad_(True)
            cost_est.backward()
            assert xt.grad is not None

            if self.args.norm is True:
                xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
            xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)

        return xt.detach()

    def test_step(self, batch, batch_idx, split='test'):
        if self.args.consistency:
            return self.consistency_trainer.consistency_test_step(self, batch, batch_idx, split)
        else:
            return self.categorical_test_step(batch, batch_idx, split)

    def categorical_test_step(self, batch, batch_idx, split='test'):
        device = batch[-1].device

        real_batch_idx, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        stacked_predict_labels = []
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        )

        if self.args.parallel_sampling > 1:
            edge_index = self.duplicate_edge_index(self.args.parallel_sampling, edge_index, node_labels.shape[0],
                                                   device)

        for _ in range(self.args.sequential_sampling):
            xt = torch.randn_like(node_labels.float())
            if self.args.parallel_sampling > 1:
                xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)
            # if self.diffusion_type == 'gaussian':
            #     xt.requires_grad = True
            # else:
            xt = (xt > 0).long().reshape(-1)

            batch_size = 1
            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                              T=self.diffusion.T, inference_T=steps)

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = torch.tensor([t1 for _ in range(batch_size)]).int()
                t2 = torch.tensor([t2 for _ in range(batch_size)]).int()

                xt = self.categorical_denoise_step(xt, t1, device, edge_index, target_t=t2)

            # if self.diffusion_type == 'gaussian':
            #     predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
            # else:
            #     predict_labels = xt.float().cpu().detach().numpy() + 1e-6
            predict_labels = xt.float().cpu().detach().numpy() + 1e-6   # 770,

            stacked_predict_labels.append(predict_labels)

        # import time
        # b = time.time()
        predict_labels = np.concatenate(stacked_predict_labels, axis=0)  # 770,
        all_sampling = self.args.sequential_sampling * self.args.parallel_sampling
        split_predict_labels = np.split(predict_labels, all_sampling)
        # print(len(split_predict_labels), split_predict_labels[0].shape)
        solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in split_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.max(solved_costs)
        best_solved_id = np.argmax(solved_costs)
        # print('pl', time.time() - b)
        # input()

        gt_cost = node_labels.cpu().numpy().sum()

        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        guided_gap, g_best_solved_cost = -1., -1.
        if self.args.rewrite:
            g_best_solution = solved_solutions[best_solved_id]
            for _ in range(3):
                g_stacked_predict_labels = []
                g_x0 = torch.from_numpy(g_best_solution).unsqueeze(0).to(device)
                g_x0 = F.one_hot(g_x0.long(), num_classes=2).float()

                steps_T = int(self.args.diffusion_steps * self.args.rewrite_ratio)
                # steps_inf = int(self.args.inference_diffusion_steps * self.args.rewrite_ratio)
                steps_inf = 10

                time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                                  T=steps_T, inference_T=steps_inf)

                Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0.device)
                g_xt_prob = torch.matmul(g_x0, Q_bar)  # [B, N, 2]
                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(g_x0.device)  # [B, N]

                if self.args.parallel_sampling > 1:
                    g_xt = g_xt.repeat(self.args.parallel_sampling, 1, 1)
                g_xt = g_xt.reshape(-1)
                # g_xt = (g_xt > 0).long().reshape(-1)
                for i in range(steps_inf):
                    t1, t2 = time_schedule(i)
                    t1 = torch.tensor([t1]).int()
                    t2 = torch.tensor([t2]).int()
                    g_xt = self.guided_categorical_denoise_step(g_xt, t1, device, edge_index, target_t=t2)

                g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
                g_stacked_predict_labels.append(g_predict_labels)
                g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)

                g_split_predict_labels = np.split(g_predict_labels, self.args.parallel_sampling)
                g_solved_solutions = [mis_decode_np(g_predict_labels, adj_mat) for g_predict_labels in
                                      g_split_predict_labels]
                g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]
                g_best_solved_cost = np.max([g_best_solved_cost, np.max(g_solved_costs)])
                g_best_solved_id = np.argmax(g_solved_costs)

                g_best_solution = g_solved_solutions[g_best_solved_id]
            # print(
            #     f'tot_points: {g_x0.shape[-2]}, gt_cost: {gt_cost}, selected_points: {best_solved_cost} -> {g_best_solved_cost}')
        if self.args.rewrite:
            metrics = {
                f"{split}/gap": gap,
                f"{split}/guided_gap": guided_gap,
                f"{split}/gt_cost": gt_cost,
                f"{split}/guided_solved_cost": g_best_solved_cost,
            }
        else:
            metrics = {
                f"{split}/gap": gap,
                f"{split}/gt_cost": gt_cost,
            }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_state_dict'] = self.model.state_dict()

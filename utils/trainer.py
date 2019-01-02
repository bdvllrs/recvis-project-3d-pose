from datetime import datetime
from tqdm import tqdm
import os
from utils import viz
from utils import data
from utils import cameras
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch

__all__ = ['Trainer']


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Trainer:
    def __init__(self, train_loader, val_loader, optimizer, model, human_dataset, log_every: int = 50,
                 save_folder: str = None):
        """
        Trainer class
        Args:
            train_loader: training loader
            val_loader: validation loader
            optimizer: optimizer
            model:
            log_every: Print log every batch
            save_folder: folder to save the learned models
        """
        self.save_folder = save_folder
        self.path = None
        self.log_every = log_every
        self.model = model
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.device = torch.device("cpu")
        self.human_dataset = human_dataset
        self.criterion = torch.nn.MSELoss(reduction='none')

        self.plot_logs = False
        if not self.plot_logs:
            plt.switch_backend('agg')

        self.logs = {
            "training_error": [],
            "testing_error": [],
            "epochs": [],
            "loss_mm": []
        }

        self.glob_step = 0

        self.lr_now = 0.001
        self.lr_decay = 100000
        self.lr_gamma = 0.96

        self.current_epoch = 0

    def to(self, device):
        self.device = device
        return self

    def step_train(self, epoch):
        self.model.train()
        self.step(self.train_loader, epoch, "train")

    def step_val(self, epoch):
        self.model.eval()
        self.step(self.val_loader, epoch, "val")

    def step(self, loader, epoch, type):
        total_loss = 0
        batch_size = loader.batch_size
        sample = np.arange(len(loader.dataset))
        np.random.shuffle(sample)
        sample = sample[:15]
        viz_samples_2d, viz_samples_pred, viz_samples_true, viz_root_positions = [], [], [], []
        viz_keys = [], [], []
        k = 0
        loss_mm_mean = []
        with tqdm(total=len(loader.dataset) / loader.batch_size) as t:
            for batch_id, data in enumerate(loader):
                data_2d, data_3d, root_position, keys = data
                data_2d, data_3d = data_2d.to(self.device, torch.float), data_3d.to(self.device, torch.float)
                root_position = root_position.to(self.device, torch.float)
                if type == "train":
                    self.optimizer.zero_grad()
                    loss, out, loss_mm = self.forward(data_2d, data_3d, type)
                    loss.backward()
                    # Clip grad
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optimizer.step()
                    if batch_id % self.log_every == 0:
                        t.set_description("Train - Epoch " + str(epoch))
                        t.set_postfix_str("Loss: " + str(loss.data.item()))
                    t.update()
                else:
                    loss, out, loss_mm = self.forward(data_2d, data_3d, type)
                    # loss_mm_mean.append(loss_mm.detach().cpu().numpy())
                for i in range(loader.batch_size):
                    if k in sample:
                        viz_samples_2d.append(data_2d.detach().cpu().numpy()[i])
                        viz_samples_pred.append(out.detach().cpu().numpy()[i])
                        viz_samples_true.append(data_3d.detach().cpu().numpy()[i])
                        viz_root_positions.append(root_position.detach().cpu().numpy()[i])
                        viz_keys[0].append(keys[0][i])
                        viz_keys[1].append(keys[1][i])
                        viz_keys[2].append(keys[2][i])
                    k += 1
                total_loss += loss.data.item()
                # get the index of the max log-probability
                if batch_id % self.log_every == 0:
                    text = "Val - Epoch" if type == "val" else "Train - Epoch"
                    t.set_description(text + str(epoch))
                    t.set_postfix_str("Loss: " + str(loss.data.item()))
                t.update()

            total_loss /= len(loader.dataset) / batch_size
            self.logs["testing_error" if type == "val" else "training_error"].append(total_loss)
            # if type == "val":
            #     self.logs["loss_mm"].append(np.mean(loss_mm_mean))
            if type == "val":
                # print('\nValidation set: Average loss:', total_loss, 'mm', self.logs["loss_mm"][-1], '\n')
                print('\nValidation set: Average loss:', total_loss, '\n')
            else:
                print('\nTraining set: Average loss:', total_loss, '\n')

            viz_samples_2d = np.array(viz_samples_2d)
            viz_samples_pred = np.array(viz_samples_pred)
            viz_samples_true = np.array(viz_samples_true)
            viz_root_positions = np.array(viz_root_positions)

            self.visualize(viz_samples_2d, viz_samples_pred, viz_samples_true, viz_root_positions, viz_keys, type)
            self.plot_learning_curves()

    def train(self, n_epochs: int):
        """
        Args:
            n_epochs: Number of epochs
        """
        if self.save_folder is not None and self.path is None:
            self.path = self.save_folder + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            os.mkdir(self.path)
            os.mkdir(self.path + "/logs")
        for epoch in range(1, n_epochs + 1):
            self.current_epoch = epoch
            self.logs["epochs"].append(epoch)
            self.step_train(epoch)
            self.step_val(epoch)
            if self.save_folder is not None:
                model_file = self.path + '/model.pth'
                torch.save(self.model.state_dict(), model_file)
            print('\nSaved models in ' + self.path + '.')

    def forward(self, data, target, type):
        out = self.model(data)
        loss = self.criterion(out, target)
        distances = None
        # if type == "val":
        #     distances = torch.zeros(loss.shape[0], loss.shape[1] // 3)
        #     for i in range(loss.shape[0]):
        #         for index, k in enumerate(range(0, loss.shape[1] // 3, 3)):
        #             distances[index] = torch.sqrt(loss[i, k:k + 3].sum())
        #     distances = distances.mean()
        return loss.mean(), out, distances

    def plot_learning_curves(self):
        if len(self.logs['training_error']) == len(self.logs['testing_error']):
            if self.plot_logs:
                fig = plt.figure()
                plt.plot(self.logs['epochs'], self.logs['training_error'], label="Training")
                plt.plot(self.logs['epochs'], self.logs['testing_error'], label="Testing")
                plt.plot(self.logs['epochs'], self.logs['loss_mm'], label="Loss mm test")
                plt.title('Learning curves')
                plt.xlabel("Epochs")
                plt.ylabel("MSE")
                plt.legend()
                plt.show()
            else:
                with open(self.path + "/logs/log.pkl", 'wb') as f:
                    pickle.dump(self.logs, f)

    def unormalize_2d_data(self, batch):
        return data.un_normalize_data(batch, self.human_dataset.data_mean_2d,
                                      self.human_dataset.data_std_2d, self.human_dataset.dim_to_ignore_2d)

    def unormalize_3d_data(self, batch):
        return data.un_normalize_data(batch, self.human_dataset.data_mean_3d,
                                      self.human_dataset.data_std_3d, self.human_dataset.dim_to_ignore_3d)

    def visualize(self, data_2d, prediction, target, root_positions, keys, type):
        """
        Plots the result
        Adapted from https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/predict_3dpose.py
        Args:
            data_2d: 2d points
            prediction: 3D predictions
            target: 3D ground truth
        """

        # Visualize random samples
        import matplotlib.gridspec as gridspec

        # 1080p	= 1,920 x 1,080
        fig = plt.figure(figsize=(19.2, 10.8))

        gs1 = gridspec.GridSpec(5, 9)  # 5 rows, 9 columns
        gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
        plt.axis('off')

        subplot_idx, exidx = 1, 0
        nsamples = len(data_2d)

        data_2d = self.unormalize_2d_data(data_2d)
        target = self.unormalize_3d_data(target)
        prediction = self.unormalize_3d_data(prediction)

        for i in np.arange(nsamples):
            key = (keys[0][exidx].item(), keys[1][exidx], keys[2][exidx])
            target[exidx], prediction[exidx] = self.flatten_to_camera(target[exidx], prediction[exidx],
                                                                      root_positions[exidx], key)
            # Plot 2d pose
            ax1 = plt.subplot(gs1[subplot_idx - 1])
            p2d = data_2d[exidx, :]
            viz.show2Dpose(p2d, ax1)
            ax1.invert_yaxis()

            # Plot 3d gt
            ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
            p3d = target[exidx, :]
            viz.show3Dpose(p3d, ax2)

            # Plot 3d predictions
            ax3 = plt.subplot(gs1[subplot_idx + 1], projection='3d')
            p3d = prediction[exidx, :]
            viz.show3Dpose(p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71")

            exidx = exidx + 1
            subplot_idx = subplot_idx + 3

        plt.savefig(self.path + "/logs/poses_" + type + "_" + str(self.current_epoch) + ".eps", type="eps", dpi=1000)
        if self.plot_logs:
            plt.show()

    def flatten_to_camera(self, target, prediction, root_positions, keys):
        N_CAMERAS = 4
        N_JOINTS_H36M = 32

        # Add global position back
        target = target + np.tile(root_positions, [1, N_JOINTS_H36M])

        # Load the appropriate camera
        subj, _, sname = keys

        cname = sname.split('.')[1]  # <-- camera name
        scams = {(subj, c + 1): self.human_dataset.cameras[(subj, c + 1)] for c in
                 range(N_CAMERAS)}  # cams of this subject
        scam_idx = [scams[(subj, c + 1)][-1] for c in range(N_CAMERAS)].index(cname)  # index of camera used
        the_cam = scams[(subj, scam_idx + 1)]  # <-- the camera used
        R, T, f, c, k, p, name = the_cam
        assert name == cname

        def cam2world_centered(data_3d_camframe):
            data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
            data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M * 3))
            # subtract root translation
            return data_3d_worldframe - np.tile(data_3d_worldframe[:, :3], (1, N_JOINTS_H36M))

        # Apply inverse rotation and translation
        target = cam2world_centered(target)

        prediction = cam2world_centered(prediction)
        return target, prediction

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, r2_score
from joblib import Parallel, delayed
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR

class MultiTaskModel(nn.Module):
    """
    A PyTorch-based Multi-Task Learning model that allows for shared layers across tasks and dedicated layers for each task.
    The model supports both binary classification and regression tasks and uses different loss functions accordingly.
    """

    def __init__(self, input_size, train_data, val_data, train_label, val_label,
                 epochs, batch_size, shared_str, dedicated_str, ntasks, binary,
                 label, benchmark = None, k=1, thresh=0.0001, duplicate=True, dropout_rate=0.3):
        """
        Initializes the MultiTaskModel4.

        Parameters:
        - input_size (int): The size of the input feature vector.
        - train_data (torch.Tensor): The training data features.
        - val_data (torch.Tensor): The validation data features.
        - test_data (torch.Tensor): The testing data features.
        - train_label (torch.Tensor): The training data labels.
        - val_label (torch.Tensor): The validation data labels.
        - test_label (torch.Tensor): The testing data labels.
        - epochs (int): The number of epochs to train the model.
        - batch_size (int): The size of each batch for training.
        - shared_str (list[int]): The structure (number of units) of the shared layers.
        - dedicated_str (list[int] or list[list[int]]): The structure (number of units) of the dedicated layers for each task. If `duplicate` is True, this should be a single list.
        - ntasks (int): The number of tasks.
        - binary (list[bool]): A list indicating whether each task is a binary classification (`True`) or a regression (`False`).
        - k (float): The parameter for the inverse quadratic switching function.
        - thresh (float): The threshold for early stopping (not used in this implementation).
        - duplicate (bool): If `True`, use the same `dedicated_str` structure for all tasks; otherwise, each task has its own dedicated structure.
        - dropout_rate (float): The dropout rate for regularization.
        """
        super(MultiTaskModel, self).__init__()

        # Validation checks
        if ntasks < 1:
            raise ValueError("Number of tasks must be at least 1.")
        if not duplicate and len(dedicated_str) != ntasks:
            raise ValueError("Length of `dedicated_str` must match `ntasks` when `duplicate` is False.")

        # Initialize parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.ntasks = ntasks
        self.thresh = thresh
        self.k = k
        self.binary = binary
        self.label = label
        self.benchmark = benchmark
        self.train_label = train_label
        self.val_label = val_label
        torch.manual_seed(1)
        np.random.seed(1)
        # Shared layers
        self.shared_layers = nn.Sequential()
        for i, units in enumerate(shared_str):
            self.shared_layers.add_module(f'linear_{i}', nn.Linear(input_size, units))
            self.shared_layers.add_module(f'relu_{i}', nn.LeakyReLU())
            self.shared_layers.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))
            input_size = units

        # Dedicated layers for each task
        self.task_layers = nn.ModuleList()
        for i in range(ntasks):
            task = nn.Sequential()
            hidden_size = input_size

            if duplicate:
                task_str = dedicated_str
            else:
                task_str = dedicated_str[i]

            for j, units in enumerate(task_str):
                task.add_module(f'linear_block{i}_{j}', nn.Linear(hidden_size, units))
                task.add_module(f'relu_block{i}_{j}', nn.LeakyReLU())
                task.add_module(f'dropout_block{i}_{j}', nn.Dropout(dropout_rate))
                hidden_size = units

            task.add_module(f'output_{i}', nn.Linear(hidden_size, 1))
            self.task_layers.append(task)


        # Prepare datasets and dataloaders
        self.train_dataset = TensorDataset(train_data, train_label)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = TensorDataset(val_data, val_label)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define loss functions
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Initialize optimizer with task-specific learning rates
        param_groups = [{'params': self.shared_layers.parameters(), 'lr': 0.001}]  # Shared layers
        for i, task in enumerate(self.task_layers):
            valid_labels = np.where(~torch.isnan(train_label[:, i]))[0]  # Handling NaN values
            valid_ratio = len(valid_labels) / len(train_label)
            task_lr = 0.001 * self.switching_function(valid_ratio) # Task-specific learning rates
            #print(i,len(valid_labels),task_lr)
            param_groups.append({'params': task.parameters(), 'lr': task_lr})

        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        # Initialize separate schedulers for each task
        #self.schedulers = [
        #    ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True, cooldown=0, threshold=1e-4)
        #    for _ in range(24)
        #]

        # Initialize separate CosineAnnealingLR schedulers for each task
        self.schedulers = [
            CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)  # T_max is the max number of iterations/epochs for annealing
            for _ in range(len(self.task_layers))
        ]
    def switching_function(self, x):
        """
        Logarithmic switching function.

        Parameters:
        - x (float): The number of valid labels for the task.

        Returns:
        - float: The scaling factor for the learning rate.
        """
        return 1 + np.log(1 + self.k*x)

    '''
    def switching_function(self, x):
        """
        Modified inverse quadratic switching function.

        Parameters:
        - x (float): The number of valid labels for the task.

        Returns:
        - float: The scaling factor for the learning rate.
        """
        return 1 + self.k * (x + 1)**2
    '''
    def compute_auc(self, y_true, y_pred):
        """
        Compute AUC score.

        Parameters:
        - y_true (torch.Tensor): True labels.
        - y_pred (torch.Tensor): Predicted scores.

        Returns:
        - float: AUC score.
        """
        y_true = y_true
        y_pred = y_pred
        if len(np.unique(y_true)) == 1:
            # Only one class present, AUC is undefined
            return 0.5  # 0.5 for random
        return roc_auc_score(y_true, y_pred)


    def compute_r2(self, y_true, y_pred):
        """
        Compute R² score.

        Parameters:
        - y_true (torch.Tensor): True labels.
        - y_pred (torch.Tensor): Predicted values.

        Returns:
        - float: R² score.
        """
        y_true = y_true
        y_pred = y_pred
        return r2_score(y_true, y_pred)


    def compute_task_loss(self, i, shared_out, y_batch, task_weights):
        task_output = self.task_layers[i](shared_out).squeeze()
        task_label = y_batch[:, i]

        valid_indices = ~torch.isnan(task_label)
        valid_task_output = task_output[valid_indices]
        valid_task_label = task_label[valid_indices]

        if self.binary[i]:
            bce_loss = self.bce_loss(valid_task_output, valid_task_label)
            if valid_task_output.numel() == 0:
                auc_score = 0.0  # or any other default value
            else:
                #bce_loss = self.bce_loss(valid_task_output, valid_task_label)
                auc_score = self.compute_auc(
                    valid_task_label.clone().cpu().detach().numpy(),
                    torch.sigmoid(valid_task_output.clone()).cpu().detach().numpy(),
                )
                auc_score = torch.tensor(auc_score, device=valid_task_output.device)

            loss = task_weights[i] * (bce_loss - auc_score)  # Adjusting weight
            #print("binary",bce_loss,auc_score)
        else:
            mse_loss = self.mse_loss(valid_task_output, valid_task_label)
            if valid_task_output.numel() < 2:
                r2_score_val = 0.0  # or any other default value
            else:

                r2_score_val = self.compute_r2(
                    valid_task_label.clone().cpu().detach().numpy(),
                    valid_task_output.clone().cpu().detach().numpy(),
                )
                r2_score_val = torch.tensor(r2_score_val, device=valid_task_output.device)

            loss = task_weights[i] *((mse_loss - r2_score_val) ) # Adjusting weight
            #print("not binary",mse_loss, r2_score_val)
        return loss/valid_task_output.numel()


    def compute_metrics(self,i):
        train_label = np.asarray(self.train_label)
        val_label = np.asarray(self.val_label)
    
        train_mask = ~np.isnan(train_label[:, i])
        val_mask = ~np.isnan(val_label[:, i])
    
        y_train = train_label[train_mask, i]
        y_val = val_label[val_mask, i]

        if self.binary[i]:
            y_train_pred = torch.sigmoid(self.train_predictions[f'task_{i}'])[~np.isnan(train_label[:, i])].cpu().detach().numpy()
            y_val_pred = torch.sigmoid(self.validation_predictions[f'task_{i}'])[~np.isnan(val_label[:, i])].cpu().detach().numpy()

            train_auc = self.compute_auc(y_train, y_train_pred)
            val_auc = self.compute_auc(y_val, y_val_pred)

            return train_auc, float('nan'), val_auc, float('nan')
        else:
            y_train_pred = self.train_predictions[f'task_{i}'][~np.isnan(train_label[:, i])].cpu().detach().numpy()
            y_val_pred = self.validation_predictions[f'task_{i}'][~np.isnan(val_label[:, i])].cpu().detach().numpy()

            train_r2 = self.compute_r2(y_train, y_train_pred)
            val_r2 = self.compute_r2(y_val, y_val_pred)

            return float('nan'), train_r2, float('nan'), val_r2


    def Fit(self):
        """
        Trains the model for the specified number of epochs.

        Returns:
        - None
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        improved_tasks = np.zeros(self.epochs)
        train_tasks = np.zeros(self.epochs)
        task_weights = torch.ones(self.ntasks).to(next(self.parameters()).device)  # Initial task weights
        #print("1")
        for epoch in range(self.epochs):
            self.train()
            train_loss = np.zeros(self.ntasks)

            for x_batch, y_batch in self.train_dataloader:
                self.optimizer.zero_grad()
                shared_out = self.shared_layers(x_batch)
                tasks = list(range(self.ntasks))
                random.shuffle(tasks)

                for i in tasks:
                    loss = self.compute_task_loss(i, shared_out, y_batch,task_weights)
                    loss.backward(retain_graph=True)
                    train_loss[i] += loss.item()

                # Add gradient noise
                for param in self.parameters():
                    if param.grad is not None:
                        noise = torch.normal(mean=0.0, std=0.01, size=param.grad.shape).to(param.grad.device)
                        param.grad += noise

                self.optimizer.step()

            total_train_loss = train_loss / len(self.train_dataloader)
            #print("training done")
            self.eval()
            val_loss = np.zeros(self.ntasks)
            #print("training done")
            with torch.no_grad():
                for x_batch, y_batch in self.val_dataloader:
                    shared_out = self.shared_layers(x_batch)
                    for i in range(self.ntasks):
                        loss = self.compute_task_loss(i, shared_out, y_batch, task_weights)
                        val_loss[i] += loss.item()
                        #print(i,val_loss[i])


                total_val_loss = val_loss / len(self.val_dataloader)

            # Make preedictions
            self.train_predictions = self.forward(self.train_dataloader.dataset.tensors[0])
            self.validation_predictions = self.forward(self.val_dataloader.dataset.tensors[0])

            # Use Parallel to compute metrics for all tasks
            #results = Parallel(n_jobs=24)(
            #    delayed(self.compute_metrics)(i) for i in range(self.ntasks)
            #)
            train_auc_track = np.zeros(self.ntasks)
            train_r2_track = np.zeros(self.ntasks)
            val_auc_track = np.zeros(self.ntasks)
            val_r2_track = np.zeros(self.ntasks)
            for i in range(self.ntasks):
                train_auc_track[i], train_r2_track[i], val_auc_track[i], val_r2_track[i] = self.compute_metrics(i)


            # Ensure values are rounded to 2 decimal places before comparison
            improved_tasks[epoch] = (
                (np.round(np.array(val_auc_track)[:-6], 2) > np.round(self.benchmark[:-6], 2)).sum() +
                (np.round(np.array(val_r2_track)[-6:], 2) > np.round(self.benchmark[-6:], 2)).sum()
            )

            train_tasks[epoch] = (
                (np.round(np.array(train_auc_track)[:-6], 2) > np.round(self.benchmark[:-6], 2)).sum() +
                (np.round(np.array(train_r2_track)[-6:], 2) > np.round(self.benchmark[-6:], 2)).sum()
            )

            # Dynamic task weighting based on validation loss
            if epoch > 0:
                task_weights = np.exp(-0.1 * total_val_loss)  # Decrease weight for lower losses

            # Step the learning rate scheduler for each task
            for i, scheduler in enumerate(self.schedulers):
                scheduler.step(total_val_loss[i])  # Each scheduler tracks its own task loss

            # Print results in tabular format
            print(f"Epoch: {epoch + 1}/{self.epochs}")# Total no of improved tasks = {improved_tasks[epoch]}")
            print(f"{'Task':<6} {'Train Loss':<12} {'Train AUC':<10} {'Train R²':<10} {'Val Loss':<12} {'Val AUC':<10} {'Val R²':<10} {'benchmark':<10}")
            for i in range(self.ntasks):
                print(f"Task {i+1:<2} {total_train_loss[i]:<12.4f} {train_auc_track[i]:<10.4f} {train_r2_track[i]:<10.4f} {total_val_loss[i]:<12.4f} {val_auc_track[i]:<10.4f} {val_r2_track[i]:<10.4f}{self.benchmark[i]:<10.4f}")
            print("Total no of improved tasks", improved_tasks[epoch])
            print("-" * 85)
            if (improved_tasks[epoch] >= 13) and (train_tasks[epoch] == 24):
                break
        return improved_tasks


    def forward(self, dataset):
        """
        Extracts the final outputs for each task from an unlabeled dataset.

        Parameters:
        - dataset (torch.Tensor): The input dataset.

        Returns:
        - task_outputs (dict): A dictionary where each key is a task identifier ('task_0', 'task_1', etc.)
                            and the corresponding value is a list of outputs for that task.
        """
        shared_out = self.shared_layers(dataset)
        task_outputs = {}

        for i, task in enumerate(self.task_layers):
            task_output = task(shared_out).squeeze()
            task_outputs[f'task_{i}'] = task_output

        return task_outputs

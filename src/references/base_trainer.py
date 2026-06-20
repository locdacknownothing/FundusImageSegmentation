import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=100, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_performance = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, performance):
        improved = False
        if self.mode == 'min':
            # Specifically for HD95/loss, ignore 0.0 results, require strictly better
            if performance > 1e-8 and performance < self.best_performance:
                self.best_performance = performance
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else:
            if performance > self.best_performance:
                self.best_performance = performance
                self.counter = 0
                improved = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            
        return improved
        
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_performance': self.best_performance,
            'early_stop': self.early_stop
        }
        
    def load_state_dict(self, state_dict):
        self.counter = state_dict.get('counter', 0)
        self.best_performance = state_dict.get('best_performance', float('inf') if self.mode == 'min' else float('-inf'))
        self.early_stop = state_dict.get('early_stop', False)


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        optimizer: torch.optim.Optimizer,
        criterion,
        writer: SummaryWriter,
        val_dataset=None,
        patience: int = 100,
        early_stopping_mode: str = 'max',
        early_stopping_metric: str = 'val_metric',
        snapshot_path: str = 'checkpoints',
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        device=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.patience = patience
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_metric = early_stopping_metric
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parse datasets
        if isinstance(dataset, dict):
            self.train_dataset = dataset.get('train')
            self.val_dataset = dataset.get('val', val_dataset)
        elif isinstance(dataset, (list, tuple)) and len(dataset) == 2:
            self.train_dataset = dataset[0]
            self.val_dataset = dataset[1]
        else:
            self.train_dataset = dataset
            self.val_dataset = val_dataset
            
        self.early_stopping = EarlyStopping(patience=self.patience, mode=self.early_stopping_mode)
        self.iter_num = 0
        self.start_epoch = 0
        
        # Initialize DataLoaders
        self.train_loader = self.get_dataloader(self.train_dataset, shuffle=True)
        if self.val_dataset is not None:
            self.val_loader = self.get_dataloader(self.val_dataset, shuffle=False)
        else:
            self.val_loader = None

    def get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.get_worker_init_fn()
        )

    def get_worker_init_fn(self):
        return None

    def process_batch(self, batch):
        """Extracts inputs and targets from the batch. Subclasses can override if needed."""
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
            return inputs, targets, {}
        elif isinstance(batch, dict):
            inputs = batch['image']
            targets = batch['label']
            extra_info = {k: v for k, v in batch.items() if k not in ['image', 'label']}
            return inputs, targets, extra_info
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def compute_loss(self, outputs, targets, extra_info):
        """Computes training loss. Subclasses can override if they have complex loss definitions."""
        return self.criterion(outputs, targets)

    def on_train_iter_end(self, epoch, batch_idx, loss_val, inputs, targets, outputs, extra_info):
        """Hook called at the end of each training iteration."""
        pass

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, sampled_batch in enumerate(self.train_loader):
            inputs, targets, extra_info = self.process_batch(sampled_batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.compute_loss(outputs, targets, extra_info)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            self.iter_num += 1
            
            # Save results to writer at every iteration
            self.writer.add_scalar('train/loss_iter', loss.item(), self.iter_num)
            
            self.on_train_iter_end(epoch, batch_idx, loss.item(), inputs, targets, outputs, extra_info)
            
        avg_epoch_loss = epoch_loss / len(self.train_dataset)
        return avg_epoch_loss

    def val_epoch(self, epoch):
        """Validates 1 epoch. Must return the metric used for early stopping, or a dict containing it."""
        raise NotImplementedError("Subclasses must implement val_epoch.")

    def on_epoch_end(self, epoch, train_loss, val_result):
        """Hook called at the end of each epoch (after validation)."""
        pass

    def save_checkpoint(self, epoch, filename):
        os.makedirs(self.snapshot_path, exist_ok=True)
        save_path = os.path.join(self.snapshot_path, filename)
        
        state = {
            'epoch': epoch + 1,
            'iter_num': self.iter_num,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_performance': self.early_stopping.best_performance,
            'early_stopping': self.early_stopping.state_dict()
        }
        torch.save(state, save_path)
        self.log_checkpoint(epoch, filename)

    def log_checkpoint(self, epoch, filename):
        pass

    def resume(self, resume_path):
        if os.path.isdir(resume_path):
            latest_file = os.path.join(resume_path, 'latest_model.pth')
            if os.path.exists(latest_file):
                resume_file = latest_file
            else:
                resume_file = os.path.join(resume_path, 'best_model.pth')
        else:
            resume_file = resume_path

        if os.path.exists(resume_file):
            checkpoint = torch.load(resume_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            self.iter_num = checkpoint['iter_num']
            if 'early_stopping' in checkpoint:
                self.early_stopping.load_state_dict(checkpoint['early_stopping'])
            elif 'best_performance' in checkpoint:
                self.early_stopping.best_performance = checkpoint['best_performance']
            print(f"Resumed from {resume_file} at epoch {self.start_epoch}")
        else:
            print(f"Checkpoint file {resume_file} not found. Starting training from scratch.")

    def train(self, max_epochs, resume_path=None):
        if resume_path:
            self.resume(resume_path)
            
        for epoch in range(self.start_epoch, max_epochs):
            train_loss = self.train_epoch(epoch)
            
            # Validate epoch
            val_result = self.val_epoch(epoch)
            
            # Determine early stopping metric
            if isinstance(val_result, dict):
                perf_metric = val_result.get(self.early_stopping_metric, 0.0)
            else:
                perf_metric = val_result
                
            self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
            
            # Check early stopping
            improved = self.early_stopping(perf_metric)
            if improved:
                self.save_checkpoint(epoch, filename='best_model.pth')
                
            self.save_checkpoint(epoch, filename='latest_model.pth')
            
            self.on_epoch_end(epoch, train_loss, val_result)
            
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

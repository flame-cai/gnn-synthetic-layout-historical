# training/main_train_eval.py
import yaml
import logging
from pathlib import Path
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import wandb
import argparse

# Local imports
# from ..gnn_data_preparation.config_models import DatasetCreationConfig
from ..gnn_data_preparation.dataset_generator import HistoricalLayoutGNNDataset
from ..gnn_data_preparation.utils import setup_logging
from .utils import set_seed, get_device, save_checkpoint, calculate_class_weights, create_prediction_log, FocalLoss
from .models.gnn_models import get_gnn_model
from .models.sklearn_models import get_sklearn_model
from .engine import train_one_epoch, evaluate
from .metrics import calculate_metrics
from .visualization import visualize_graph_predictions

logging.basicConfig(
    filename="train.log",       # file to save logs
    filemode="w",               # overwrite each run, use "a" to append
    level=logging.INFO,         # minimum level to log
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- EarlyStopping Class Definition ---
class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, mode='max',
                 checkpoint_path=None, save_checkpoint_fn=None, save_checkpoints_enabled=True):
        """
        Args:
            patience (int): How many epochs to wait after last validation metric improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
                            Default: 0
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity
                        monitored has stopped decreasing; in 'max' mode it will stop when the
                        quantity monitored has stopped increasing. Default: 'max'.
            checkpoint_path (Path): Path where to save the best model checkpoint. Required if save_checkpoint_fn
                                    is provided and save_checkpoints_enabled is True.
            save_checkpoint_fn (callable): The function to call to save the checkpoint.
                                          Expected signature: `func(model, optimizer, epoch, val_metrics, path)`.
            save_checkpoints_enabled (bool): If False, the early stopper will not save any checkpoints,
                                             but still track the best score and trigger early stopping.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.save_checkpoint_fn = save_checkpoint_fn
        self.save_checkpoints_enabled = save_checkpoints_enabled

        if self.mode == 'min':
            self.val_score_multiplier = -1
        elif self.mode == 'max':
            self.val_score_multiplier = 1
        else:
            raise ValueError(f"Mode {self.mode} not supported. Must be 'min' or 'max'.")

        if self.save_checkpoints_enabled and (not self.save_checkpoint_fn or not self.checkpoint_path):
            logging.warning("EarlyStopping initialized with save_checkpoints_enabled=True, but save_checkpoint_fn or checkpoint_path is missing. Checkpoints will NOT be saved.")
            self.save_checkpoints_enabled = False # Disable saving if arguments are incomplete

    def __call__(self, current_metric_value, model, optimizer, epoch, val_metrics_dict_for_save):
        """
        Call this method after each validation epoch.
        Args:
            current_metric_value (float): The current value of the monitored metric (e.g., val_f1_score).
            model: The model to save.
            optimizer: The optimizer to save.
            epoch: The current epoch number.
            val_metrics_dict_for_save (dict): Full dictionary of validation metrics to save in checkpoint.
                                              This will be passed to save_checkpoint_fn.
        """
        # Flip sign if 'max' mode to make it a minimization problem for consistent comparison
        score = self.val_score_multiplier * current_metric_value 
        logging.info(f'The precision is: {current_metric_value}')

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, val_metrics_dict_for_save)
        elif score < self.best_score + self.delta: 
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Improved
            if self.verbose:
                logging.info(f'Validation metric improved ({self.val_score_multiplier * self.best_score:.6f} --> {current_metric_value:.6f}). Saving model ...')
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, val_metrics_dict_for_save)
            self.counter = 0

    def _save_checkpoint(self, model, optimizer, epoch, val_metrics_dict_for_save):
        """Saves model when validation metric improves, if saving is enabled."""
        if self.save_checkpoints_enabled:
            self.save_checkpoint_fn(model, optimizer, epoch, val_metrics_dict_for_save, self.checkpoint_path)
        elif self.verbose:
            logging.debug("Skipping model save as save_checkpoints is disabled in config.")


def run_gnn_fold(config, fold_dir, model_name, run_dir):
    """Trains and evaluates a GNN model for a single fold."""
    device = get_device(config['device'])
    
    # 1. Load Data
    train_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'train'))
    val_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'val'))
    test_dataset = HistoricalLayoutGNNDataset(root=str(fold_dir / 'gnn' / 'test'))

    
    train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True,num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'], shuffle=False,num_workers=0, pin_memory=False)

    # 2. Initialize Model, Optimizer, Loss
    model = get_gnn_model(model_name, config, train_dataset[0]).to(device)
    optimizer = getattr(torch.optim, config['training_params']['optimizer'])(model.parameters(), lr=config['training_params']['learning_rate'])
    
    # --- START OF CHANGE: Learning Rate Warmup Scheduler ---
    warmup_epochs = config['training_params'].get('lr_warmup_epochs', 0)
    scheduler = None
    if warmup_epochs > 0:
        def warmup_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linearly increase the learning rate multiplier from a small fraction to 1
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                # After warmup, the multiplier is 1 (i.e., use the base_lr)
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        logging.info(f"Using learning rate warmup for {warmup_epochs} epochs.")
    # --- END OF CHANGE ---

    if config['training_params']['imbalance_handler'] == 'weighted_loss':
        weights = calculate_class_weights(train_dataset).to(device)
        logging.info(f"Using weighted cross-entropy with weights: {weights}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    elif config['training_params']['imbalance_handler'] == 'focal_loss':
        logging.info("Using Focal Loss")
        loss_fn = FocalLoss(alpha=config['training_params']['focal_loss_alpha'], gamma=config['training_params']['focal_loss_gamma'])
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        
    # --- Early Stopping Setup ---
    # Extract early stopping parameters from config, with sensible defaults
    early_stopping_patience = config['training_params'].get('early_stopping_patience', 10)
    early_stopping_min_delta = config['training_params'].get('early_stopping_min_delta', 0.0001)
    early_stopping_mode = config['training_params'].get('early_stopping_mode', 'max') # 'max' for metrics like F1, 'min' for loss

    # The metric to monitor for early stopping is the same as the checkpoint metric.
    # We need to extract the raw metric name from the config, e.g., 'val_f1_score' -> 'f1_score'.
    monitor_metric_key = config['checkpoint_metric'].replace('val_', '') 
    
    # Initialize EarlyStopping, passing the save_checkpoint function
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True, # Set to True to see messages about improvements and counters
        delta=early_stopping_min_delta,
        mode=early_stopping_mode,
        checkpoint_path=run_dir / 'best_model.pt',
        save_checkpoint_fn=save_checkpoint, # Pass the existing save_checkpoint utility
        save_checkpoints_enabled=config['save_checkpoints'] # Respect the global config for saving
    )

    logging.info("--- PRE-TRAINING LOADER TEST ---")
    try:
        # We will use the *exact same loader* that crashes later
        test_batch = next(iter(val_loader)) 
        logging.info(">>> SUCCESS: next(iter(val_loader)) worked perfectly before training. <<<")
        # It's a good idea to reset the iterator, though it will be re-created in the evaluate function
        val_loader_iterator = iter(val_loader)
    except Exception as e:
        logging.error(f"FATAL: val_loader crashed even BEFORE training. Error: {e}")

    # Now, start your normal training loop
    logging.info("--- Starting Training Loop ---")

    # 3. Training Loop
    for epoch in range(1, config['training_params']['epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{config['training_params']['epochs']} ---")
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch number {epoch}")
        val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device)
        print("evaluation done")
        log_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        log_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # --- START OF CHANGE: Log LR and Step Scheduler ---
        current_lr = optimizer.param_groups[0]['lr']
        log_metrics['learning_rate'] = current_lr
        
        logging.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Textline F1: {val_metrics['textline_f1_score']:.4f}, LR: {current_lr:.6f}")
        
        if scheduler:
            scheduler.step()
        # --- END OF CHANGE ---

        if config['tracking']['use_tracker'] == 'wandb':
            wandb.log(log_metrics, step=epoch)

        # --- Early Stopping Check and Checkpoint Saving ---
        current_metric_for_es = val_metrics[monitor_metric_key]
        early_stopper(current_metric_for_es, model, optimizer, epoch, val_metrics)

        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch} due to no improvement in '{monitor_metric_key}' for {early_stopping_patience} epochs.")
            break # Exit the training loop
    
    # 4. Final Evaluation on Test Set
    logging.info("--- Final Evaluation on Test Set ---")
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location=device)
    model = checkpoint['model'] # <-- LOAD THE MODEL OBJECT (As requested)

    # --- START OF CHANGE ---
    # Load original textline labels into a dictionary keyed by page_id
    # true_textline_labels_by_page = {}
    # data_creation_config_path = Path(config['dataset_path']).parent.parent /'configs/1_dataset_creation_config.yaml'
    # # This logic assumes the Phase 1 config is available to find the raw data dir
    # # A more robust way would be to save this info in the dataset itself
    # with open(data_creation_config_path, 'r') as f:
    #     d_config_dict = yaml.safe_load(f)
        
    # raw_data_dir = Path(d_config_dict['input_data_dir'])
    
    # for data in test_dataset:
    #     try:
    #         label_path = raw_data_dir / f"{data.page_id}_labels_textline.txt"
    #         true_textline_labels_by_page[data.page_id] = np.loadtxt(label_path, dtype=int)
    #     except Exception as e:
    #         logging.warning(f"Could not load textline labels for page {data.page_id}: {e}")
            
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device)

    print("evaluate function run")

    log_test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    logging.info(f"Test Metrics: {log_test_metrics}")
    if config['tracking']['use_tracker'] == 'wandb':
        wandb.log(log_test_metrics)

    
    
    # 5. Save artifacts
    if config['save_prediction_log']:
        pred_df = create_prediction_log(test_dataset, test_preds, test_labels, model_name, fold_dir.name.split('_')[-1])
        pred_df.to_csv(run_dir / 'test_predictions.csv', index=False)

    if config['generate_visualizations']:
        pred_idx = 0
        for i, data in enumerate(test_dataset):
            print(f"visualizing {i}")
            if i >= config['num_visualizations']: break
            num_edges = data.edge_index.shape[1]
            page_preds = test_preds[pred_idx : pred_idx + num_edges]
            viz_path = run_dir / 'visualizations' / f"page_{data.page_id}.png"
            visualize_graph_predictions(data.cpu(), page_preds, viz_path)
            pred_idx += num_edges
            
    return log_test_metrics

def run_sklearn_fold(config, fold_dir, model_name, run_dir):
    """Trains and evaluates a scikit-learn model for a single fold."""
    # 1. Load Data
    train_df = pd.read_csv(fold_dir / 'sklearn' / 'train.csv')
    test_df = pd.read_csv(fold_dir / 'sklearn' / 'test.csv')

    X_train = train_df.drop(columns=['label', 'page_id', 'source_node_id', 'target_node_id'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label', 'page_id', 'source_node_id', 'target_node_id'])
    y_test = test_df['label']

    # 2. Train Model
    model = get_sklearn_model(model_name, config)
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # 3. Evaluate
    logging.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred) # Sklearn models don't have graph-level metrics here
    log_metrics = {f"test_{k}": v for k, v in metrics.items()}
    
    logging.info(f"Test Metrics for {model_name}: {log_metrics}")
    if config['tracking']['use_tracker'] == 'wandb':
         wandb.log(log_metrics)
         
    return log_metrics


def main():
    """Main execution function to run model training and cross-validation."""
    # --- NEW: Setup command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Run model training and cross-validation.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/2_training_config.yaml',
        help='Path to the YAML configuration file for training.'
    )
    parser.add_argument('--dataset_path', type=str, help='Override the dataset path from the config file.')
    parser.add_argument('--output_dir', type=str, help='Override the output directory from the config file.')
    parser.add_argument('--unique_folder_name', type=str, help='name of the unique folder specifiying datasize and grouping for this run.')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='Specific GPU index to use (e.g., 0, 1). If provided, overrides the device setting in the config file.'
    )
    args = parser.parse_args()

    # Load configuration from the path specified in the arguments
    config_path = Path(args.config)
    if not config_path.exists():
        logging.critical(f"Configuration file not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['random_seed'])

    # --- MODIFIED: Prioritize command-line arguments over config values ---
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(config['dataset_path'])
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output_dir'])
    
    # --- START OF CHANGE 2: Override config['device'] based on --gpu_id ---
    if args.gpu_id is not None:
        if args.gpu_id < torch.cuda.device_count():
            # Set the device string to 'cuda:X'
            config['device'] = f'cuda:{args.gpu_id}'
            logging.info(f"Overriding device to: {config['device']} from command line argument --gpu_id.")
        else:
             logging.warning(f"GPU ID {args.gpu_id} requested, but only {torch.cuda.device_count()} available. Using config default or falling back.")
             
    # Ensure a sensible default if the config itself is missing the device key (optional, but good practice)
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- END OF CHANGE 2 ---


    folds_dir = dataset_path / "folds"
    if not folds_dir.exists():
        logging.critical(f"Folds directory not found at {folds_dir}")
        return
        
    all_results = []

    for model_name in config['models_to_run']:
        model_type = config['model_configs'][model_name]['type']
        
        for fold_dir in sorted(folds_dir.glob('fold_*')):
            fold_idx = int(fold_dir.name.split('_')[-1])
            logging.info(f"===== Starting Fold {fold_idx} for Model: {model_name} =====")
            
            # Setup run directory and logging, now using the potentially overridden output_dir
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = config['tracking']['run_name_template'].format(model_name=model_name, fold_idx=fold_idx, timestamp=timestamp)
            run_dir = output_dir / args.unique_folder_name / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(run_dir / 'training.log')
            
            # Initialize tracker
            if config['tracking']['use_tracker'] == 'wandb':
                wandb.init(
                    project=config['project_name'],
                    name=run_name,
                    config=config,
                    reinit=True # Important for loops
                )
            
            if model_type == 'gnn':
                results = run_gnn_fold(config, fold_dir, model_name, run_dir)
            elif model_type == 'sklearn':
                results = run_sklearn_fold(config, fold_dir, model_name, run_dir)
            else:
                raise ValueError(f"Unknown model type {model_type}")
            
            results['model'] = model_name
            results['fold'] = fold_idx
            all_results.append(results)
            
            if config['tracking']['use_tracker'] == 'wandb':
                wandb.finish()

    # Aggregate and save final results, using the potentially overridden output_dir
    results_df = pd.DataFrame(all_results)
    agg_results = results_df.groupby('model').agg(['mean', 'std']).reset_index()
    
    logging.info("\n\n===== Aggregated Cross-Validation Results =====")
    print(agg_results)
    
    # Ensure the base output directory exists before saving summary files
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_results.to_csv(output_dir / 'aggregated_results.csv', index=False)
    results_df.to_csv(output_dir / 'all_fold_results.csv', index=False)
    
if __name__ == "__main__":
    main()
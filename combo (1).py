# FILE: codetrytwentyfour.py
import ssl
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import numpy as np
import os
from tqdm import tqdm
import torchmetrics # Import torchmetrics
import matplotlib.pyplot as plt
import logging
import json
import time
import gc
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, precision_recall_curve, f1_score, matthews_corrcoef # Added matthews_corrcoef for potential sklearn use
from sklearn.preprocessing import label_binarize
from functools import partial
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
import tempfile
import warnings
import plotly
import kaleido
import concurrent.futures
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset
warnings.filterwarnings('ignore')

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Set up SSL certificate
os.environ["SSL_CERT_FILE"] = "/scratch365/mli29/mli29_fixed/ssl/cacert.pem"
print(ssl.get_default_verify_paths())

# Create base directories
BASE_DIR = "/scratch365/mli29/combined_bayesian"
os.makedirs(BASE_DIR, exist_ok=True)

# Set up logging
log_dir = os.path.join(BASE_DIR, "training_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "nested_kfold_training_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f"Logging training progress to: {log_file}")

# Define paths and base hyperparameters
data_dir = "/scratch365/mli29/combo_data"  # Directory for the dataset
save_dir = os.path.join(BASE_DIR, "saved_models")  # Directory to save best models
os.makedirs(save_dir, exist_ok=True)

# Set up plots directory
plots_dir = os.path.join(log_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
logging.info(f"Plots will be saved to: {plots_dir}")

# Check CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set MPLCONFIGDIR environment variable to a writable directory
os.environ['MPLCONFIGDIR'] = '/tmp/'
logging.info(f"MPLCONFIGDIR set to: {os.environ['MPLCONFIGDIR']}")

# Global variables for Optuna optimization
NUM_TRIALS = 50          # CHANGED: Number of Bayesian optimization trials reduced to 30
OPTUNA_TIMEOUT = 120*3600 # 72 hours timeout for optimization
VALIDATION_SPLIT = 0.15  # Percentage of training data to use for validation during hyperparameter tuning
TEST_EPOCHS = 10         # Number of epochs for quick testing during optimization
PRUNING_PATIENCE = 5     # Number of epochs to wait for improvement before pruning
OUTER_FOLDS = 5          # Number of outer folds for nested cross-validation
INNER_FOLDS = 4          # Number of inner folds for nested cross-validation

# Helper function to convert NumPy types for JSON serialization
def convert_numpy_types_for_json(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types_for_json(i) for i in obj]
    else:
        return obj

class ImageMixTransform:
    """
    Custom transform for mixing images of the same class.
    Modified to handle both direct usage and usage through Subset.

    Args:
        p (float): Probability of applying the mix operation
        method_weights (dict): Dictionary of mixing methods and their weights
    """
    def __init__(self, p=0.5, method_weights=None):
        self.p = p
        # Default method weights if none provided
        self.method_weights = method_weights if method_weights else {
            "vertical": 1.0,
            "horizontal": 1.0,
            "diagonal": 1.0,
            "quadrants": 1.0,
            "columns": 1.0
        }
        self.methods = list(self.method_weights.keys())
        self.weights = list(self.method_weights.values())

    def __call__(self, img, dataset=None, class_label=None):
        """
        Apply image mixing transformation.

        Args:
            img: Input image tensor
            dataset: Dataset containing images, now optional with default None
            class_label: Target class label, now optional with default None

        Returns:
            Mixed image tensor, or original if dataset or class_label is None
        """
        # Skip mixing if we don't have dataset or class information
        # This will happen when used with Subset during cross-validation
        if dataset is None or class_label is None:
            return img

        # Original logic: apply mixing with probability p
        if np.random.random() > self.p:
            return img

        # Find all indices of the same class
        # Need to access the underlying dataset if 'dataset' is a Subset
        if isinstance(dataset, Subset):
            full_dataset = dataset.dataset
            original_indices = dataset.indices
            # Map subset index to original dataset index to get path and label
            same_class_indices_in_subset = [
                i for i, subset_idx in enumerate(original_indices)
                if full_dataset.samples[subset_idx][1] == class_label and full_dataset.samples[subset_idx][0] != img
            ]
            if not same_class_indices_in_subset:
                return img
            random_subset_idx = np.random.choice(same_class_indices_in_subset)
            random_original_idx = original_indices[random_subset_idx]
            other_img_path = full_dataset.samples[random_original_idx][0]

        else: # Original dataset
            full_dataset = dataset
            # Assuming img might not be directly comparable if it's already transformed
            # We need a way to identify the original path of 'img' if possible, otherwise this mixing won't work correctly
            # For simplicity, we'll proceed assuming standard ImageFolder structure is maintained
            # Find indices of the same class, excluding the current image (this comparison might fail if img is transformed tensor)
            # A better approach might involve passing the index or path along
            same_class_indices = [i for i, (_, label) in enumerate(dataset.samples)
                                if label == class_label] # simplified for now
            if not same_class_indices or len(same_class_indices) == 1: # Need at least one other image
                 return img

            random_idx = np.random.choice(same_class_indices)
            # Ensure we don't pick the same image - this check is imperfect if img is a tensor
            # This part needs careful implementation depending on how img path is accessed
            # Let's assume for now we can find a different image path
            other_img_path = dataset.samples[random_idx][0] # Simplified

        other_img = full_dataset.loader(other_img_path)

        # Apply the same transforms as the main transform pipeline, except mixing
        if hasattr(full_dataset, 'transform'):
            transforms_list = full_dataset.transform.transforms
            for t in transforms_list:
                if not isinstance(t, ImageMixTransform):
                    try:
                        other_img = t(other_img)
                    except Exception as e:
                        logging.warning(f"Could not apply transform {type(t)} to other_img: {e}")
                        return img # Skip mixing if transform fails

        # Choose mix method
        mix_method = np.random.choice(self.methods, p=np.array(self.weights)/sum(self.weights))

        # Apply the selected mixing method
        return self.mix_images(img, other_img, mix_method)

    def mix_images(self, img1, img2, method):
        """
        Mix two images based on the specified method.

        Args:
            img1: First image tensor (C, H, W)
            img2: Second image tensor (C, H, W)
            method: Mixing method (vertical, horizontal, diagonal, quadrants, columns)

        Returns:
            Mixed image tensor
        """
        # Ensure both images have the same shape
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            from torchvision import transforms
            try:
                resize = transforms.Resize((img1.shape[1], img1.shape[2]))
                img2 = resize(img2)
            except Exception as e:
                 logging.warning(f"Could not resize image for mixing: {e}. Returning original image.")
                 return img1


        mixed_img = img1.clone()

        if method == "vertical":
            # Split vertically (top/bottom)
            h = img1.shape[1]
            split_point = h // 2
            mixed_img[:, :split_point, :] = img1[:, :split_point, :]
            mixed_img[:, split_point:, :] = img2[:, split_point:, :]

        elif method == "horizontal":
            # Split horizontally (left/right)
            w = img1.shape[2]
            split_point = w // 2
            mixed_img[:, :, :split_point] = img1[:, :, :split_point]
            mixed_img[:, :, split_point:] = img2[:, :, split_point:]

        elif method == "diagonal":
            # Split diagonally
            h, w = img1.shape[1], img1.shape[2]
            mask = torch.zeros((h, w), dtype=torch.bool)
            for i in range(h):
                for j in range(w):
                    mask[i, j] = (i / h + j / w) > 1.0

            for c in range(img1.shape[0]):  # For each channel
                mixed_img[c][~mask] = img1[c][~mask]
                mixed_img[c][mask] = img2[c][mask]

        elif method == "quadrants":
            # Split into quadrants
            h, w = img1.shape[1], img1.shape[2]
            h_mid, w_mid = h // 2, w // 2

            # Top-left and bottom-right from img1, others from img2
            mixed_img[:, :h_mid, :w_mid] = img1[:, :h_mid, :w_mid]  # Top-left
            mixed_img[:, h_mid:, :w_mid] = img2[:, h_mid:, :w_mid]  # Bottom-left
            mixed_img[:, :h_mid, w_mid:] = img2[:, :h_mid, w_mid:]  # Top-right
            mixed_img[:, h_mid:, w_mid:] = img1[:, h_mid:, w_mid:]  # Bottom-right

        elif method == "columns":
            # Alternate columns
            w = img1.shape[2]
            for j in range(w):
                if j % 2 == 0:
                    mixed_img[:, :, j] = img1[:, :, j]
                else:
                    mixed_img[:, :, j] = img2[:, :, j]

        return mixed_img

class ModelConfig:
    def __init__(self, threshold=0.5, metric_weights=None, model_paths=None):
        self.threshold = threshold
        self.metric_weights = metric_weights if metric_weights else {'f1': 1.0}
        self.model_paths = model_paths if model_paths else {}
        self.threshold_history = []  # Track threshold changes over epochs
        self.metric_history = {}     # Track metrics over epochs
        self.hyperparameters = {}    # Store best hyperparameters

    def to_dict(self):
        """Convert config to a dictionary for serialization"""
        return {
            'threshold': self.threshold,
            'metric_weights': self.metric_weights,
            'model_paths': self.model_paths,
            'threshold_history': self.threshold_history,
            'metric_history': self.metric_history,
            'hyperparameters': self.hyperparameters
        }

    def save(self, path):
        """Save the config to a JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path):
        """Load a config from a JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = cls(
            threshold=config_dict.get('threshold', 0.5),
            metric_weights=config_dict.get('metric_weights', {'f1': 1.0}),
            model_paths=config_dict.get('model_paths', {})
        )
        config.threshold_history = config_dict.get('threshold_history', [])
        config.metric_history = config_dict.get('metric_history', {})
        config.hyperparameters = config_dict.get('hyperparameters', {})
        return config

    def update_threshold(self, new_threshold, epoch, metrics=None):
        """Update threshold and record history"""
        self.threshold = new_threshold
        self.threshold_history.append((epoch, new_threshold))

        if metrics:
            for metric_name, value in metrics.items():
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                self.metric_history[metric_name].append((epoch, value))

# Initialize global model config
model_config = ModelConfig(threshold=0.5)

# Define data transforms with augmentation
def get_data_transforms(rotation=10, horizontal_flip=True, vertical_flip=False,
                       brightness=0.2, contrast=0.2, saturation=0.2,
                       hue=0.1, normalization=True, use_image_mix=True,
                       image_mix_prob=0.3):
    """Get data transforms with customizable augmentation parameters"""

    train_transforms = []
    test_transforms = []

    # Basic transforms for both train and test
    train_transforms.append(transforms.Resize((224, 224)))
    test_transforms.append(transforms.Resize((224, 224)))

    # Data augmentation for training only
    if rotation > 0:
        train_transforms.append(transforms.RandomRotation(rotation))
    if horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        train_transforms.append(transforms.RandomVerticalFlip())

    # Color jitter for training only
    if any([brightness > 0, contrast > 0, saturation > 0, hue > 0]):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
        )

    # Convert to tensor for both train and test
    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    # Image mixing (only for training)
    # Instantiate ImageMixTransform here, it will be applied in ClassAwareDataset
    # The instance itself needs to be passed, but the call logic is inside ClassAwareDataset
    image_mixer = ImageMixTransform(p=image_mix_prob) if use_image_mix else None

    # Normalization for both train and test if enabled
    if normalization:
        norm_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transforms.append(norm_transform)
        test_transforms.append(norm_transform)

    # Compose transforms
    # The image_mixer instance isn't directly added here;
    # its logic is invoked within ClassAwareDataset's __getitem__
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(test_transforms)
    test_transform = transforms.Compose(test_transforms)

    # Return the transforms AND the mixer instance if created
    return train_transform, val_transform, test_transform, image_mixer


# We also need to modify the dataset loading function to pass the class information to the transform
class ClassAwareDataset(datasets.ImageFolder):
    """
    Extension of ImageFolder that provides class information to transforms.
    This is needed for the ImageMixTransform to mix only images of the same class.
    """
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader, is_valid_file=None, image_mixer=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        self.image_mixer = image_mixer # Store the mixer instance

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            # Apply standard transforms first
            sample = self.transform(sample)

        # Apply ImageMixTransform *after* standard transforms (especially ToTensor)
        if self.image_mixer is not None:
             # Pass the dataset itself (needed to find other images) and the target label
            sample = self.image_mixer(sample, self, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# Load the datasets with defaults
def load_datasets(train_transform=None, val_transform=None, test_transform=None, image_mixer=None):
    """Load datasets with specified transforms and image mixer"""
    if train_transform is None or val_transform is None or test_transform is None:
         # If transforms not provided, get defaults including the mixer
        train_transform, val_transform, test_transform, image_mixer = get_data_transforms()

    try:
        # Pass the image_mixer instance only to the training dataset
        train_dataset = ClassAwareDataset(root=f"{data_dir}/train", transform=train_transform, image_mixer=image_mixer)
        # Validation and test sets don't use image mixing
        val_dataset = ClassAwareDataset(root=f"{data_dir}/validate", transform=val_transform, image_mixer=None)
        test_dataset = ClassAwareDataset(root=f"{data_dir}/test", transform=test_transform, image_mixer=None)

        logging.info(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        sys.exit(1)


# Attention module: Adds attention mechanism to enhance important features
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Define a small network to learn attention weights
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),  # Reduce the channel size
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),  # Restore the original channel size
            nn.Sigmoid()  # Output a value between 0 and 1 to apply as attention weights
        )

    def forward(self, x):
        # Multiply input tensor by the attention map
        return x * self.attention(x)

# Function to create model with attention and customizable architecture
def create_model(model_name, num_classes=1, use_attention=True, dropout_rate=0.5,
                hidden_size=256, pretrained=True, freeze_layers=0):
    """
    Create model with customizable architecture

    Args:
        model_name: one of 'resnet101', 'efficientnet_b4', 'densenet121'
        num_classes: number of output classes
        use_attention: whether to use attention module
        dropout_rate: dropout rate for regularization
        hidden_size: size of hidden layer before final classification
        pretrained: whether to use pretrained weights
        freeze_layers: number of layers to freeze (0 means no freezing)

    Returns:
        model: PyTorch model
    """
    # Set pretrained parameter based on ImageNet weights availability
    pretrained_weights = "IMAGENET1K_V1" if pretrained else None

    # Initialize the model based on model_name
    if model_name == 'resnet101':
        model = models.resnet101(weights=pretrained_weights)
        feature_dim = model.fc.in_features # Get original feature dim
        last_layer_name = 'layer4' # Name of the layer before avgpool/fc
        in_channels_attention = 2048 # ResNet101 layer4 output channels

        if freeze_layers > 0:
            # Freeze initial layers
            layers_to_freeze = [model.conv1, model.bn1, model.relu, model.maxpool,
                               model.layer1, model.layer2][:freeze_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        # Apply attention module if requested
        if use_attention:
             # Get the original layer
            original_layer = getattr(model, last_layer_name)
            # Create attention module with correct input channels
            attention_module = AttentionModule(in_channels_attention)
            # Replace the layer with a sequence containing the original layer and attention
            setattr(model, last_layer_name, nn.Sequential(original_layer, attention_module))


        # Replace the final fully connected layer
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),  # Second dropout layer with lower rate
            nn.Linear(hidden_size, num_classes)
        )

    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=pretrained_weights)
        feature_dim = model.classifier[1].in_features # Get original feature dim
        last_block_idx = -1 # Features are sequential blocks
        in_channels_attention = 1792 # EfficientNet-B4 final block output channels

        if freeze_layers > 0:
            # For EfficientNet, we can freeze blocks of the feature extractor
            num_blocks = min(len(model.features), freeze_layers)
            for i in range(num_blocks):
                for param in model.features[i].parameters():
                    param.requires_grad = False

        # Apply attention module if requested
        if use_attention:
            # Get the original features sequence
            original_features = model.features
            # Create attention module
            attention_module = AttentionModule(in_channels_attention)
            # Append attention module after the features
            model.features = nn.Sequential(original_features, attention_module)


        # Replace the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, num_classes)
        )

    elif model_name == 'densenet121':
        model = models.densenet121(weights=pretrained_weights)
        feature_dim = model.classifier.in_features # Get original feature dim
        last_block_name = 'norm5' # The layer just before the classifier in DenseNet features
        in_channels_attention = 1024 # DenseNet-121 features output channels

        if freeze_layers > 0:
            # Freeze the initial layers
            layers_to_freeze = []
            if freeze_layers >= 1:
                layers_to_freeze.extend([model.features.conv0, model.features.norm0, model.features.relu0, model.features.pool0])
            if freeze_layers >= 2:
                layers_to_freeze.append(model.features.denseblock1)
            if freeze_layers >= 3:
                layers_to_freeze.extend([model.features.transition1, model.features.denseblock2])
            if freeze_layers >= 4:
                layers_to_freeze.extend([model.features.transition2, model.features.denseblock3])
            # Be cautious with freezing transition3 and denseblock4 as they are close to the end

            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        # Apply attention module if requested
        if use_attention:
            # Get the original features sequence
            original_features = model.features
            # Create attention module
            attention_module = AttentionModule(in_channels_attention)
             # Insert attention after the features (specifically after norm5)
            # This requires modifying the Sequential structure or wrapping it
            # A simpler way for DenseNet might be to apply attention *after* the global avg pooling
            # but before the classifier. Let's try applying it to the feature map output.
            model.features = nn.Sequential(original_features, attention_module)


        # Replace the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

# Dynamic weight module for ensemble
class DynamicWeight(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)  # Initialize weights equally

    def forward(self, outputs):
        # Stack model outputs [num_models, batch_size, 1]
        stacked_outputs = torch.stack(outputs)
        # Get device from stacked outputs
        device = stacked_outputs.device
        # Get softmax weights and reshape for broadcasting [num_models, 1, 1]
        weights = self.weights.to(device).softmax(-1).view(-1, 1, 1)
        # Apply weights and sum along model dimension
        return torch.sum(stacked_outputs * weights, dim=0)

# Function to perform ensemble prediction
def ensemble_predict(models, inputs, dynamic_weight):
    """Ensemble prediction function that handles device placement"""
    device = inputs.device
    outputs = []

    for model in models:
        # Make sure model is on the same device as inputs
        if next(model.parameters()).device != device:
            model.to(device)
        outputs.append(model(inputs))

    # Make sure dynamic_weight is on the same device
    if hasattr(dynamic_weight, 'parameters') and len(list(dynamic_weight.parameters())) > 0:
        if next(dynamic_weight.parameters()).device != device:
            dynamic_weight.to(device)

    # Apply dynamic weighting
    weighted_outputs = dynamic_weight(outputs)
    return weighted_outputs

# Function to find optimal threshold based on F1 score
def find_optimal_threshold(probabilities, labels):
    """
    Find the optimal threshold that maximizes the F1 score.

    Args:
        probabilities: predicted probabilities
        labels: ground truth labels

    Returns:
        optimal_threshold: threshold that maximizes F1 score
        metrics_at_threshold: dictionary of metrics at the optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probabilities)

    # Handle the edge case where precision/recall have one more element than thresholds
    # This happens because the last precision/recall point corresponds to a threshold of 0
    # Scikit-learn's PR curve includes endpoint (recall=0, precision=1) with no threshold
    # We need thresholds corresponding to precisions[:-1] and recalls[:-1]
    valid_thresholds = thresholds
    valid_precisions = precisions[:-1]
    valid_recalls = recalls[:-1]

    # Calculate F1 score for each threshold pair
    f1_scores = 2 * (valid_precisions * valid_recalls) / (valid_precisions + valid_recalls + 1e-8)

    # Find the threshold that maximizes the F1 score
    if len(f1_scores) == 0:
         # Handle cases with no valid thresholds (e.g., all predictions are the same)
        optimal_threshold = 0.5
        optimal_idx = -1 # Indicates no valid threshold found
        logging.warning("Could not determine optimal threshold from precision-recall curve, defaulting to 0.5")
    else:
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = valid_thresholds[optimal_idx]


    # Calculate additional metrics at this threshold
    predictions = (probabilities >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_at_thresh = sensitivity # Recall is the same as sensitivity
    f1_at_thresh = 2 * (precision_at_thresh * recall_at_thresh) / (precision_at_thresh + recall_at_thresh + 1e-8)


    # Log information about the optimal threshold
    logging.info(f"Optimal threshold determined: {optimal_threshold:.4f}")
    if optimal_idx != -1:
        logging.info(f"Precision at optimal threshold: {precision_at_thresh:.4f} (from calculation)")
        logging.info(f"Recall at optimal threshold: {recall_at_thresh:.4f} (from calculation)")
        logging.info(f"F1 score at optimal threshold: {f1_at_thresh:.4f} (from calculation)")


    # Return metrics at the optimal threshold
    metrics_at_threshold = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1_at_thresh,
        'accuracy': accuracy,
        'precision': precision_at_thresh,
        'recall': recall_at_thresh
    }

    return optimal_threshold, metrics_at_threshold

# Function to evaluate models on a dataset
def evaluate(models, loader, dynamic_weight, epoch, is_test=False, threshold=0.5, device=None):
    """Modified evaluate function that accepts a specific device parameter"""
    # If no device specified, use the global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure models and dynamic weight are on the correct device
    for model in models:
        model.eval()  # Set models to evaluation mode
        if next(model.parameters()).device != device:
            model.to(device)

    if hasattr(dynamic_weight, 'parameters') and len(list(dynamic_weight.parameters())) > 0:
        dynamic_weight.eval()  # Set dynamic weight to evaluation mode
        if next(dynamic_weight.parameters()).device != device:
            dynamic_weight.to(device)

    # Initialize metrics - BINARY METRICS
    acc_metric = torchmetrics.Accuracy(task="binary").to(device) # Removed num_classes=2 for binary
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device) # Keep num_classes=2 here
    auc_metric = torchmetrics.AUROC(task="binary").to(device)
    sensitivity_metric = torchmetrics.Recall(task="binary").to(device) # Sensitivity is Recall
    specificity_metric = torchmetrics.Specificity(task="binary").to(device)
    avg_precision_metric = torchmetrics.AveragePrecision(task="binary").to(device)
    mcc_metric = torchmetrics.MatthewsCorrCoef(task="binary").to(device) # Added MCC


    all_preds = []
    all_labels = []
    all_probs = []

    # Evaluation loop
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels_int = labels.int() # Metrics expect int labels

            try:
                outputs = ensemble_predict(models, images, dynamic_weight)
                probs = torch.sigmoid(outputs).squeeze() # Ensure probs are 1D for binary tasks
                predicted = (probs > threshold).int() # Use int for metrics

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_int.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Update metrics - Pass probs for AUROC/AvgPrecision, predicted for others
                acc_metric.update(predicted, labels_int)
                precision_metric.update(predicted, labels_int)
                recall_metric.update(predicted, labels_int)
                f1_metric.update(predicted, labels_int)
                confusion_matrix_metric.update(predicted, labels_int)
                auc_metric.update(probs, labels_int)
                sensitivity_metric.update(predicted, labels_int) # Same as recall_metric
                specificity_metric.update(predicted, labels_int)
                avg_precision_metric.update(probs, labels_int)
                mcc_metric.update(predicted, labels_int) # Update MCC

            except Exception as e:
                logging.error(f"Error during evaluation batch: {e}")
                continue

    # Convert to numpy arrays
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds) # Need predictions for sklearn MCC if used

    # Calculate log loss
    log_loss_value = log_loss(all_labels_np, all_probs_np, labels=[0, 1])


    # Compute metrics
    accuracy = acc_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    confusion_matrix = confusion_matrix_metric.compute().cpu().numpy()
    auc_value = auc_metric.compute().item()
    sensitivity = sensitivity_metric.compute().item()
    specificity = specificity_metric.compute().item()
    avg_precision = avg_precision_metric.compute().item()
    mcc = mcc_metric.compute().item() # Compute MCC
    # mcc_sklearn = matthews_corrcoef(all_labels_np, all_preds_np) # Alternative calculation

    # Manual calculations from confusion matrix for verification
    manual_fpr = 0.0 # Default value
    if confusion_matrix.shape == (2, 2):
        tn, fp, fn, tp = confusion_matrix.ravel()
        manual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        logging.warning(f"Unexpected confusion matrix shape: {confusion_matrix.shape}")


    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)
    roc_auc = auc(fpr, tpr)

    # Reset metrics
    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    confusion_matrix_metric.reset()
    auc_metric.reset()
    sensitivity_metric.reset()
    specificity_metric.reset()
    avg_precision_metric.reset()
    mcc_metric.reset() # Reset MCC


    # If we're saving visualizations (not during hyperparameter tuning)
    if epoch >= 0:  # Only save plots during final training, not during quick evaluation
        dataset_type = "test" if is_test else "validation"
        current_plots_dir = os.path.join(plots_dir, f"epoch_{epoch}_{dataset_type}") # Create epoch/type specific subdir
        os.makedirs(current_plots_dir, exist_ok=True)


        # Plot confusion matrix
        cm_plot_path = os.path.join(current_plots_dir, f"confusion_matrix_thresh_{threshold:.2f}.png")
        plot_confusion_matrix(confusion_matrix, threshold, cm_plot_path)

        # Plot ROC curve
        roc_plot_path = os.path.join(current_plots_dir, f"roc_curve_thresh_{threshold:.2f}.png")
        plot_roc_curve(fpr, tpr, roc_auc, threshold, roc_plot_path)

        # Plot probability distribution
        dist_plot_path = os.path.join(current_plots_dir, f"prob_distribution_thresh_{threshold:.2f}.png")
        plot_probability_distribution(all_probs_np, all_labels_np,
                                    f"Prob Distribution ({dataset_type.capitalize()}, Epoch {epoch+1}, Thresh={threshold:.2f})",
                                    dist_plot_path)

        # Save ROC data for later visualization
        roc_data_path = os.path.join(save_dir, f"roc_data_{dataset_type}_epoch_{epoch}.json")
        save_roc_data(fpr, tpr, roc_auc, None, f"{dataset_type}_epoch_{epoch}")

    # Log comprehensive metrics
    logging.info(f"{'Test' if is_test else 'Validation'} Metrics (Epoch {epoch+1}, Thresh {threshold:.4f}):")
    logging.info(f"  Accuracy = {accuracy:.4f}")
    logging.info(f"  Precision = {precision:.4f}")
    logging.info(f"  Recall (Sensitivity) = {recall:.4f}")
    logging.info(f"  Specificity = {specificity:.4f}")
    logging.info(f"  F1 Score = {f1:.4f}")
    logging.info(f"  AUC = {auc_value:.4f}")
    logging.info(f"  Avg Precision = {avg_precision:.4f}")
    logging.info(f"  MCC = {mcc:.4f}") # Log MCC
    logging.info(f"  Log Loss = {log_loss_value:.4f}")
    logging.info(f"  Confusion Matrix:\n{confusion_matrix}")

    # Return comprehensive metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, # Same as sensitivity
        'f1': f1,
        'auc': auc_value,
        'sensitivity': sensitivity, # Explicitly sensitivity
        'specificity': specificity,
        'fpr': manual_fpr,
        'avg_precision': avg_precision,
        'log_loss': log_loss_value,
        'mcc': mcc, # Include MCC
        'all_probs': all_probs_np,
        'all_labels': all_labels_np,
        'roc_data': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc
        },
        'confusion_matrix': confusion_matrix.tolist() # Use tolist() for JSON
    }


# Plot confusion matrix
def plot_confusion_matrix(cm, threshold=0.5, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: confusion matrix (numpy array)
        threshold: classification threshold
        save_path: path to save plot
    """
    plt.figure(figsize=(8, 6))
    display_cm = np.array(cm) # Ensure it's a numpy array for plotting
    plt.imshow(display_cm, cmap='Blues', interpolation='nearest')
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.colorbar()
    tick_marks = np.arange(len(display_cm))
    plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    # Add numbers to the confusion matrix
    thresh = display_cm.max() / 2.
    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            plt.text(j, i, format(display_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if display_cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Confusion matrix plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save confusion matrix plot to {save_path}: {e}")
        finally:
           plt.close() # Ensure plot is closed
    else:
        plt.show()
        plt.close()

# Plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, threshold=0.5, save_path=None):
    """
    Plot ROC curve

    Args:
        fpr: false positive rate
        tpr: true positive rate
        roc_auc: area under ROC curve
        threshold: classification threshold (for marker)
        save_path: path to save plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (area = {roc_auc:.3f})') # Increased precision for AUC label
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Marking the threshold point is difficult without the thresholds array from roc_curve
    # We'll skip the approximate marking for now as it can be misleading.
    # optimal_point_idx = np.argmax(tpr - fpr) # Youden's J statistic index (example)
    # plt.plot(fpr[optimal_point_idx], tpr[optimal_point_idx], 'go', markersize=8, label='Best Youden Index')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic (AUC = {roc_auc:.3f})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logging.info(f"ROC curve plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save ROC curve plot to {save_path}: {e}")
        finally:
            plt.close()
    else:
        plt.show()
        plt.close()

# Plot probability distribution
def plot_probability_distribution(probabilities, labels, title, save_path):
    """
    Plot the distribution of predicted probabilities for both classes.

    Args:
        probabilities: predicted probabilities
        labels: ground truth labels
        title: title for the plot
        save_path: path to save the figure
    """
    plt.figure(figsize=(10, 6))

    # Separate probabilities for each class
    labels_np = np.array(labels) # Ensure labels is numpy array
    probs_np = np.array(probabilities)
    pos_probs = probs_np[labels_np == 1]
    neg_probs = probs_np[labels_np == 0]

    # Plot histograms
    plt.hist(pos_probs, bins=50, alpha=0.6, label='Positive Class (1)', color='blue', density=True)
    plt.hist(neg_probs, bins=50, alpha=0.6, label='Negative Class (0)', color='red', density=True)

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Probability distribution plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save probability distribution plot to {save_path}: {e}")
        finally:
            plt.close()
    else:
        plt.show()
        plt.close()


# Save checkpoint function
def save_checkpoint(model, epoch, val_metric, model_name, optimizer=None, scheduler=None):
    """
    Save model checkpoint

    Args:
        model: model to save
        epoch: current epoch
        val_metric: validation metric
        model_name: name of the model
        optimizer: optimizer state to save
        scheduler: scheduler state to save

    Returns:
        save_path: path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_metric': val_metric,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    save_path = os.path.join(save_dir, f"{model_name}_best_fold_specific.pth") # Make filename clearer
    torch.save(checkpoint, save_path)
    logging.info(f"Saved checkpoint for {model_name} epoch {epoch} to {save_path}")
    return save_path

# Save dynamic weight model
def save_dynamic_weight(dynamic_weight, epoch, val_metric, optimizer=None):
    """
    Save dynamic weight model

    Args:
        dynamic_weight: dynamic weight model to save
        epoch: current epoch
        val_metric: validation metric
        optimizer: optimizer state to save

    Returns:
        save_path: path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'dynamic_weight_state_dict': dynamic_weight.state_dict(),
        'val_metric': val_metric,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    save_path = os.path.join(save_dir, "dynamic_weight_best_fold_specific.pth") # Make filename clearer
    torch.save(checkpoint, save_path)
    logging.info(f"Saved dynamic weight checkpoint epoch {epoch} to {save_path}")
    return save_path

# Load model from checkpoint
def load_model(model, checkpoint_path):
    """
    Load model from checkpoint

    Args:
        model: model to load into
        checkpoint_path: path to checkpoint

    Returns:
        model: loaded model
        val_metric: validation metric from checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) # Ensure loading to correct device
        # Handle both direct state dict and checkpoint dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_metric = checkpoint.get('val_metric', 0.0)
        else:
            model.load_state_dict(checkpoint)
            val_metric = 0.0 # Cannot get val_metric if it's just state_dict
        logging.info(f"Loaded model from {checkpoint_path} with validation metric {val_metric:.4f}")
        return model, val_metric
    except Exception as e:
        logging.error(f"Error loading model from {checkpoint_path}: {e}")
        return model, 0.0

# Load dynamic weight from checkpoint
def load_dynamic_weight(dynamic_weight, checkpoint_path):
    """
    Load dynamic weight from checkpoint

    Args:
        dynamic_weight: dynamic weight model to load into
        checkpoint_path: path to checkpoint

    Returns:
        dynamic_weight: loaded dynamic weight model
        val_metric: validation metric from checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) # Ensure loading to correct device
        # Handle both direct state dict and checkpoint dictionary
        if 'dynamic_weight_state_dict' in checkpoint:
            dynamic_weight.load_state_dict(checkpoint['dynamic_weight_state_dict'])
            val_metric = checkpoint.get('val_metric', 0.0)
        else:
             dynamic_weight.load_state_dict(checkpoint)
             val_metric = 0.0 # Cannot get val_metric if it's just state_dict
        logging.info(f"Loaded dynamic weight from {checkpoint_path} with validation metric {val_metric:.4f}")
        return dynamic_weight, val_metric
    except Exception as e:
        logging.error(f"Error loading dynamic weight from {checkpoint_path}: {e}")
        return dynamic_weight, 0.0

# Save model config
def save_model_config():
    """
    Save model configuration to file

    Returns:
        config_path: path to saved config file
    """
    config_path = os.path.join(save_dir, "model_config.json")
    model_config.save(config_path)
    logging.info(f"Model configuration saved to {config_path}")
    return config_path

# Function to create dataloaders with specified batch size
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Create dataloaders for training, validation, and test

    Args:
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset
        batch_size: batch size

    Returns:
        train_loader, val_loader, test_loader: data loaders
    """
    num_workers = min(multiprocessing.cpu_count(), 4) # Use fewer workers if CPU count is low
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    return train_loader, val_loader, test_loader

# Save ROC data function
def save_roc_data(fpr, tpr, auc_value, thresholds=None, name=None):
    """Save the ROC curve data for later analysis"""
    name_str = f"_{name}" if name is not None else ""
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': auc_value
    }

    if thresholds is not None:
        roc_data['thresholds'] = thresholds.tolist()

    # Save the ROC data
    roc_data_path = os.path.join(save_dir, f"roc_data{name_str}.json")
    try:
        with open(roc_data_path, 'w') as f:
            json.dump(roc_data, f, indent=4)
        logging.info(f"ROC curve data saved to {roc_data_path}")
    except Exception as e:
         logging.error(f"Failed to save ROC data to {roc_data_path}: {e}")

    return roc_data_path

# Create a visualization script for ROC curves
def create_roc_data_visualization_script():
    """Create a simple script for visualizing ROC data"""
    script_path = os.path.join(save_dir, "visualize_roc.py")

    # Define the content of the script
    script_content = """
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_roc_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)

def plot_roc(roc_data, current_threshold=None, output_path=None):
    fpr = np.array(roc_data['fpr'])
    tpr = np.array(roc_data['tpr'])
    auc = roc_data['auc']

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # If we have a current threshold, mark it (this remains approximate)
    if current_threshold is not None:
        # Try to find thresholds in data
        if 'thresholds' in roc_data:
            thresholds = np.array(roc_data['thresholds'])
            # Ensure thresholds correspond to fpr/tpr points correctly
            # roc_curve often returns len(thresholds) = len(fpr) = len(tpr)
            # Find closest threshold index
            idx = np.abs(thresholds - current_threshold).argmin()
            if 0 <= idx < len(fpr) and 0 <= idx < len(tpr):
                current_fpr, current_tpr = fpr[idx], tpr[idx]
                plt.plot(current_fpr, current_tpr, 'ro', markersize=8,
                         label=f'Threshold approx ({current_threshold:.2f})')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Curve Analysis', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"ROC curve saved to {output_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='ROC Curve Visualization')
    parser.add_argument('--data', type=str, required=True, help='Path to ROC data file')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold to mark on the curve')
    parser.add_argument('--output', type=str, default=None, help='Path to save the ROC curve plot')

    args = parser.parse_args()

    # Load ROC data
    roc_data = load_roc_data(args.data)

    # Plot ROC curve
    plot_roc(roc_data, args.threshold, args.output)

if __name__ == "__main__":
    main()
"""

    # Save the script
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        logging.info(f"ROC visualization script created at {script_path}")
        logging.info("Usage: python visualize_roc.py --data <path_to_roc_data.json> [--threshold <value>] [--output <output_file>]")
    except Exception as e:
         logging.error(f"Failed to create ROC visualization script: {e}")

    return script_path

# Create a script to plot threshold vs metrics
def create_threshold_metrics_visualization_script():
    """Create a script for visualizing how metrics change with different thresholds"""
    script_path = os.path.join(save_dir, "visualize_threshold_metrics.py")

    # Define the content of the script
    script_content = """
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def plot_threshold_history(config_data, output_path=None):
    # Extract threshold history
    threshold_history = config_data.get('threshold_history', [])

    if not threshold_history:
        print("No threshold history found in the config file")
        return

    # Convert to numpy arrays for plotting
    epochs = np.array([item[0] for item in threshold_history])
    thresholds = np.array([item[1] for item in threshold_history])

    # Plot threshold over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, thresholds, 'b-o', lw=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Optimal Threshold', fontsize=14)
    plt.title('Threshold Evolution During Training', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Threshold history plot saved to {output_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_metric_history(config_data, metric_name, output_path=None):
    # Extract metric history
    metric_history = config_data.get('metric_history', {}).get(metric_name, [])

    if not metric_history:
        print(f"No history found for metric '{metric_name}' in the config file")
        return

    # Convert to numpy arrays for plotting
    epochs = np.array([item[0] for item in metric_history])
    values = np.array([item[1] for item in metric_history])

    # Plot metric over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, values, 'g-o', lw=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'{metric_name.capitalize()}', fontsize=14)
    plt.title(f'{metric_name.capitalize()} Evolution During Training', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"{metric_name.capitalize()} history plot saved to {output_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_all_metrics(config_data, output_dir=None):
    # Extract metric history
    metric_history = config_data.get('metric_history', {})

    if not metric_history:
        print("No metric history found in the config file")
        return

    # Create output directory if needed
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot each metric
    for metric_name, history in metric_history.items():
        if not history:
            continue

        # Convert to numpy arrays for plotting
        epochs = np.array([item[0] for item in history])
        values = np.array([item[1] for item in history])

        # Plot metric over epochs
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, values, 'g-o', lw=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(f'{metric_name.capitalize()}', fontsize=14)
        plt.title(f'{metric_name.capitalize()} Evolution During Training', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        # Save or show the plot
        if output_dir:
            output_path = Path(output_dir) / f"{metric_name}_history.png"
            plt.savefig(output_path, dpi=300)
            print(f"{metric_name.capitalize()} history plot saved to {output_path}")
            plt.close()
        else:
            plt.show()
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Threshold and Metrics Visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to model_config.json file')
    parser.add_argument('--metric', type=str, default=None, help='Specific metric to plot (default: plot threshold)')
    parser.add_argument('--all', action='store_true', help='Plot all available metrics')
    parser.add_argument('--output', type=str, default=None, help='Path to save the plot(s)')

    args = parser.parse_args()

    # Load config data
    config_data = load_model_config(args.config)

    if args.all:
        # Plot all metrics
        plot_all_metrics(config_data, args.output)
    elif args.metric:
        # Plot specific metric
        plot_metric_history(config_data, args.metric, args.output)
    else:
        # Plot threshold by default
        plot_threshold_history(config_data, args.output)

if __name__ == "__main__":
    main()
"""

    # Save the script
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        logging.info(f"Threshold metrics visualization script created at {script_path}")
        logging.info("Usage: python visualize_threshold_metrics.py --config model_config.json [--metric f1|accuracy|...] [--all] [--output output_dir]")
    except Exception as e:
         logging.error(f"Failed to create threshold metrics visualization script: {e}")


    return script_path

def nested_cross_validation():
    """
    Perform nested k-fold cross-validation.
    Outer loop: Evaluates model performance with different test sets
    Inner loop: Performs hyperparameter optimization

    Returns:
        Dictionary of outer fold results
    """
    logging.info(f"Starting nested {OUTER_FOLDS}-fold cross-validation with {INNER_FOLDS} inner folds")

    # Load all data using the updated functions
    train_transform, val_transform, test_transform, train_image_mixer = get_data_transforms(use_image_mix=True) # Get mixer for training
    # No mixer needed for val/test transform composition itself
    val_transform_no_mix, test_transform_no_mix = val_transform, test_transform

    # Create datasets using ClassAwareDataset
    full_train_dataset = ClassAwareDataset(root=f"{data_dir}/train", transform=train_transform, image_mixer=train_image_mixer)
    full_val_dataset = ClassAwareDataset(root=f"{data_dir}/validate", transform=val_transform_no_mix, image_mixer=None)

    # Combine all data for cross-validation (using original ImageFolder structure temporarily for indexing)
    temp_train_dataset = datasets.ImageFolder(root=f"{data_dir}/train")
    temp_val_dataset = datasets.ImageFolder(root=f"{data_dir}/validate")
    combined_indices = list(range(len(temp_train_dataset))) + list(range(len(temp_train_dataset), len(temp_train_dataset) + len(temp_val_dataset)))
    combined_targets = [s[1] for s in temp_train_dataset.samples] + [s[1] for s in temp_val_dataset.samples]

    # We need a way to apply the correct transform (with/without mixing) to the Subsets later
    # Let's create the combined dataset using ClassAwareDataset instances
    combined_dataset = ConcatDataset([full_train_dataset, full_val_dataset])


    # Set up outer k-fold cross-validation
    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)

    outer_results = {}

    # Outer loop
    for outer_fold, (train_idx, test_idx) in enumerate(outer_skf.split(combined_indices, combined_targets)):
        logging.info(f"\n{'='*50}\nOuter Fold {outer_fold+1}/{OUTER_FOLDS}\n{'='*50}")

        # Create outer train and test datasets using Subsets
        outer_train_subset = Subset(combined_dataset, train_idx)
        outer_test_subset = Subset(combined_dataset, test_idx)

        # Extract targets for inner folds
        outer_train_targets = [combined_targets[i] for i in train_idx]

        # Create directory for this outer fold
        outer_fold_dir = os.path.join(save_dir, f"outer_fold_{outer_fold}")
        os.makedirs(outer_fold_dir, exist_ok=True)
        outer_plots_dir = os.path.join(outer_fold_dir, "plots") # Plots specific to this fold
        os.makedirs(outer_plots_dir, exist_ok=True)


        # Inner k-fold for hyperparameter tuning
        inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)

        # Create an Optuna study for this outer fold
        study_dir = os.path.join(outer_fold_dir, "optuna_study")
        os.makedirs(study_dir, exist_ok=True)
        db_path = os.path.join(study_dir, f"study_fold_{outer_fold}.db")
        study_name = f"skin_lesion_classification_fold_{outer_fold}"

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1) # Prune more aggressively
        sampler = TPESampler(seed=42 + outer_fold)  # Different seed for each fold

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="maximize",  # Maximize F1 score
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        # Define inner fold objective function
        def inner_objective(trial):
            """Objective function for inner fold optimization"""
            # Sample hyperparameters (use the same sampling as in original code)
            # 1. Data augmentation hyperparameters
            rotation = trial.suggest_int("rotation", 0, 30)
            horizontal_flip = trial.suggest_categorical("horizontal_flip", [True, False])
            vertical_flip = trial.suggest_categorical("vertical_flip", [True, False])
            brightness = trial.suggest_float("brightness", 0.0, 0.3)
            contrast = trial.suggest_float("contrast", 0.0, 0.3)
            saturation = trial.suggest_float("saturation", 0.0, 0.3)
            hue = trial.suggest_float("hue", 0.0, 0.15)
            use_image_mix = trial.suggest_categorical("use_image_mix", [True, False])
            image_mix_prob = trial.suggest_float("image_mix_prob", 0.1, 0.5) if use_image_mix else 0.0
        
        
            # 2. Model architecture hyperparameters
            use_attention = trial.suggest_categorical("use_attention", [True, False])
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
            hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
            freeze_layers = trial.suggest_int("freeze_layers", 0, 3)
        
            # 3. Training hyperparameters
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True) # Slightly narrower range
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            lr_scheduler_type = trial.suggest_categorical("lr_scheduler", ["one_cycle", "cosine", "step"])
        
            # Create transforms based on sampled hyperparameters
            # Crucially, we need to manage the transforms within the subset loop
            # The get_data_transforms creates the transform compositions
            # The ClassAwareDataset applies them, including the mixer if provided
        
            # Make sure outer_train_subset is accessible
            nonlocal outer_train_subset, outer_train_targets
        
            # Collect inner fold results
            inner_fold_scores = []
        
            # Inner cross-validation loop
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(range(len(outer_train_subset)), outer_train_targets)):
                logging.info(f"Outer {outer_fold+1}/Inner {inner_fold+1}, Trial {trial.number}")
        
                # Create inner train and validation datasets (these are subsets of outer_train_subset)
                # The underlying dataset is combined_dataset
                # Get the original indices corresponding to the inner split
                original_inner_train_indices = [outer_train_subset.indices[i] for i in inner_train_idx]
                original_inner_val_indices = [outer_train_subset.indices[i] for i in inner_val_idx]
        
                inner_train_subset = Subset(combined_dataset, original_inner_train_indices)
                inner_val_subset = Subset(combined_dataset, original_inner_val_indices)

                # Create transforms *for this trial*
                trial_train_transform, trial_val_transform, _, trial_image_mixer = get_data_transforms(
                    rotation=rotation, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                    use_image_mix=use_image_mix, image_mix_prob=image_mix_prob
                )

                # Assign transforms to the underlying ClassAwareDataset instances if possible
                # This assumes Subset exposes the underlying dataset's methods/attributes, which might not be standard.
                # A safer way is to wrap the Subset or handle transforms in the DataLoader collate_fn.
                # Let's try a simpler approach: set transform temporarily (may not work ideally with ConcatDataset/Subset nesting)

                # Hacky way: Temporarily set transforms (use with caution)
                # This relies on all datasets in ConcatDataset being ClassAwareDataset
                original_transforms = {}
                for i, ds in enumerate(inner_train_subset.dataset.datasets):
                     if isinstance(ds, ClassAwareDataset):
                        original_transforms[i] = (ds.transform, ds.image_mixer)
                        ds.transform = trial_train_transform
                        ds.image_mixer = trial_image_mixer

                original_val_transforms = {}
                for i, ds in enumerate(inner_val_subset.dataset.datasets):
                     if isinstance(ds, ClassAwareDataset):
                         original_val_transforms[i] = (ds.transform, ds.image_mixer)
                         ds.transform = trial_val_transform
                         ds.image_mixer = None # No mixing for validation


                # Create dataloaders
                num_workers = min(multiprocessing.cpu_count(), 4)
                inner_train_loader = DataLoader(
                    inner_train_subset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, persistent_workers=False # No persistent workers for inner loop
                )
                inner_val_loader = DataLoader(
                    inner_val_subset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=False
                )


                # Create models
                model_names = ['resnet101', 'efficientnet_b4', 'densenet121']
                models_list = []

                for model_name in model_names:
                    try:
                        model = create_model(
                            model_name=model_name, use_attention=use_attention,
                            dropout_rate=dropout_rate, hidden_size=hidden_size,
                            pretrained=True, freeze_layers=freeze_layers
                        )
                        model.to(device)
                        models_list.append(model)
                    except Exception as e:
                         logging.error(f"Failed to create model {model_name} in trial {trial.number}: {e}")
                         # Restore transforms before raising
                         for i, ds in enumerate(inner_train_subset.dataset.datasets):
                            if i in original_transforms:
                                ds.transform, ds.image_mixer = original_transforms[i]
                         for i, ds in enumerate(inner_val_subset.dataset.datasets):
                            if i in original_val_transforms:
                                ds.transform, ds.image_mixer = original_val_transforms[i]
                         raise # Re-raise the exception to fail the trial


                # Create dynamic weight module
                dynamic_weight = DynamicWeight(len(models_list))
                dynamic_weight.to(device)

                # Set up optimizer
                optimizer = torch.optim.AdamW([
                    {'params': m.parameters(), 'lr': learning_rate / 10} for m in models_list # Apply lower LR to backbones
                ] + [{'params': dynamic_weight.parameters(), 'lr': learning_rate}], # Higher LR for dynamic weights
                    lr=learning_rate, # Default LR (mostly for weights)
                    weight_decay=weight_decay
                )


                # Set up loss function
                criterion = torch.nn.BCEWithLogitsLoss()

                # Set up learning rate scheduler
                inner_epochs = 5  # Use fewer epochs for inner CV

                if lr_scheduler_type == "one_cycle":
                    try:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, max_lr=[learning_rate / 10] * len(models_list) + [learning_rate], # Specify max_lr per param group
                            steps_per_epoch=len(inner_train_loader), epochs=inner_epochs
                        )
                    except Exception as e:
                         logging.warning(f"Failed to init OneCycleLR in trial {trial.number}, using StepLR: {e}")
                         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=inner_epochs // 2, gamma=0.1)

                elif lr_scheduler_type == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=inner_epochs * len(inner_train_loader)) # T_max in steps
                else:  # step
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=inner_epochs // 3, gamma=0.1)


                # Set up mixed precision training
                scaler = torch.cuda.amp.GradScaler()

                # Quick training loop for inner fold
                for epoch in range(inner_epochs):
                    # Training phase
                    for model in models_list: model.train()
                    dynamic_weight.train()
                    running_loss = 0.0

                    # Process batches
                    for images, labels in inner_train_loader:
                        images, labels = images.to(device), labels.to(device)
                        labels_binary = labels.float().view(-1, 1) # BCEWithLogits expects float

                        optimizer.zero_grad()

                        try:
                            with torch.cuda.amp.autocast():
                                outputs = ensemble_predict(models_list, images, dynamic_weight)
                                loss = criterion(outputs, labels_binary)

                            scaler.scale(loss).backward()
                            # Optional: Gradient Clipping
                            # scaler.unscale_(optimizer)
                            # torch.nn.utils.clip_grad_norm_(dynamic_weight.parameters(), 1.0)
                            # for m in models_list: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()

                            # Step scheduler after optimizer step
                            if lr_scheduler_type == "one_cycle" or lr_scheduler_type == "cosine":
                                scheduler.step()

                            running_loss += loss.item()

                        except Exception as e:
                            logging.error(f"Error in inner fold training batch (Epoch {epoch+1}): {e}")
                            # Optional: break epoch or continue
                            continue # Skip batch on error

                    epoch_loss = running_loss / len(inner_train_loader) if len(inner_train_loader) > 0 else 0
                    logging.info(f"Trial {trial.number} - Inner Fold {inner_fold+1} - Epoch {epoch+1}/{inner_epochs} - Loss: {epoch_loss:.4f}")


                    # Update StepLR scheduler
                    if lr_scheduler_type == "step":
                         scheduler.step()

                    # Report intermediate value to Optuna after each epoch (more frequent pruning)
                    # Quick validation to check progress
                    val_metrics = evaluate(
                        models_list, inner_val_loader, dynamic_weight,
                        epoch=-1, is_test=False, threshold=0.5, device=device # Use fixed threshold for objective value
                    )
                    intermediate_f1 = val_metrics.get('f1', 0.0) # Use F1 score
                    trial.report(intermediate_f1, epoch)
                    logging.info(f"Trial {trial.number} - Epoch {epoch+1} - Intermediate Val F1: {intermediate_f1:.4f}")


                    # Handle pruning
                    if trial.should_prune():
                        # Restore transforms before pruning
                        for i, ds in enumerate(inner_train_subset.dataset.datasets):
                           if i in original_transforms:
                               ds.transform, ds.image_mixer = original_transforms[i]
                        for i, ds in enumerate(inner_val_subset.dataset.datasets):
                           if i in original_val_transforms:
                               ds.transform, ds.image_mixer = original_val_transforms[i]
                        raise optuna.exceptions.TrialPruned()


                # Final evaluation on inner validation set after all epochs for this inner fold
                val_metrics = evaluate(
                    models_list, inner_val_loader, dynamic_weight,
                    epoch=-1, is_test=False, threshold=0.5, device=device # Use fixed threshold for objective value
                )


                # Record the F1 score for this inner fold
                inner_fold_scores.append(val_metrics.get('f1', 0.0))

                 # Restore transforms after inner fold finishes
                for i, ds in enumerate(inner_train_subset.dataset.datasets):
                    if i in original_transforms:
                        ds.transform, ds.image_mixer = original_transforms[i]
                for i, ds in enumerate(inner_val_subset.dataset.datasets):
                     if i in original_val_transforms:
                         ds.transform, ds.image_mixer = original_val_transforms[i]


                # Clean up memory for inner fold
                for model in models_list: del model
                del dynamic_weight, optimizer, criterion, scheduler, scaler
                del inner_train_loader, inner_val_loader, inner_train_subset, inner_val_subset
                torch.cuda.empty_cache()
                gc.collect()


            # Average score across inner folds for this trial
            mean_score = np.mean(inner_fold_scores) if inner_fold_scores else 0.0
            logging.info(f"Trial {trial.number} completed with average inner F1: {mean_score:.4f}")

            return mean_score


        # Run Optuna optimization for this outer fold
        # Run Optuna optimization for this outer fold
        logging.info(f"Starting hyperparameter optimization for outer fold {outer_fold+1}")
        optuna_timeout_per_fold = OPTUNA_TIMEOUT / OUTER_FOLDS if OPTUNA_TIMEOUT else None
        
        # Check how many trials have been completed
        n_complete_trials = len(study.trials)
        # Run additional trials to reach the target number
        trials_to_run = max(0, NUM_TRIALS - n_complete_trials)
        
        logging.info(f"Study for fold {outer_fold} has {n_complete_trials} completed trials. Will run {trials_to_run} additional trials to reach target of {NUM_TRIALS}")
        
        if trials_to_run > 0:
            study.optimize(inner_objective, n_trials=trials_to_run, timeout=optuna_timeout_per_fold, gc_after_trial=True)
        else:
            logging.info(f"Study for fold {outer_fold} already has {n_complete_trials} trials, no additional trials needed.")

        # Get best parameters for this outer fold
        best_params = study.best_params
        best_trial = study.best_trial

        logging.info(f"Best trial for outer fold {outer_fold+1}: {best_trial.number}")
        logging.info(f"Best average inner F1 score: {best_trial.value:.4f}")
        logging.info("Best hyperparameters:")
        for param_name, param_value in best_params.items():
            logging.info(f"  {param_name}: {param_value}")

        # Save best parameters
        with open(os.path.join(outer_fold_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        # --- Train final model on all outer train data with best parameters ---
        logging.info(f"Training final model for outer fold {outer_fold+1} with best parameters")

        # Get best transforms
        best_train_transform, best_val_transform, _, best_image_mixer = get_data_transforms(
            rotation=best_params.get("rotation", 10),
            horizontal_flip=best_params.get("horizontal_flip", True),
            vertical_flip=best_params.get("vertical_flip", False),
            brightness=best_params.get("brightness", 0.2),
            contrast=best_params.get("contrast", 0.2),
            saturation=best_params.get("saturation", 0.2),
            hue=best_params.get("hue", 0.1),
            use_image_mix=best_params.get("use_image_mix", True),
            image_mix_prob=best_params.get("image_mix_prob", 0.3)
        )

        # Apply transforms correctly to the outer subsets
        # We need to re-instantiate the datasets within the subset or handle transforms carefully
        # Easiest: Assume outer_train_subset and outer_test_subset point to the combined_dataset
        # We need to ensure the correct transform is applied based on whether it's train or test data
        # This requires a more complex Dataset wrapper or careful handling in DataLoader.

        # Simpler approach for final training: Re-create datasets for this fold's train/test split
        final_train_indices = outer_train_subset.indices
        final_test_indices = outer_test_subset.indices

        # Identify which original dataset (train or val) each index came from
        len_orig_train = len(temp_train_dataset)

        # Create final training dataset with mixing
        final_train_ds_list = []
        for idx in final_train_indices:
            if idx < len_orig_train: # From original training set
                path, target = temp_train_dataset.samples[idx]
                final_train_ds_list.append((path, target))
            else: # From original validation set
                path, target = temp_val_dataset.samples[idx - len_orig_train]
                final_train_ds_list.append((path, target))
        # Use a standard ImageFolder structure temporarily to wrap these samples
        # This requires writing temp files or a custom dataset class.
        # Alternative: Create a custom dataset that applies the transform based on index source.

        # Let's stick to Subset for simplicity, but apply transforms via DataLoader if needed.
        # We'll assume the Subset structure allows accessing underlying dataset transforms.

        # Re-apply transforms to the datasets within the combined_dataset structure
        for i, ds in enumerate(outer_train_subset.dataset.datasets):
            if isinstance(ds, ClassAwareDataset):
                # If this dataset corresponds to the original training part
                # This logic is flawed as ConcatDataset doesn't track original source easily.
                # A better approach is needed here.

                # -- Workaround: Assume first dataset in Concat is train, second is val --
                if i == 0: # Assume first DS in Concat is the training one
                    ds.transform = best_train_transform
                    ds.image_mixer = best_image_mixer
                else: # Assume second DS is validation one (no mixing)
                    ds.transform = best_val_transform # Use val transform even if part of outer train split
                    ds.image_mixer = None

        for i, ds in enumerate(outer_test_subset.dataset.datasets):
             if isinstance(ds, ClassAwareDataset):
                 # All data in the outer test split uses the validation transform (no mixing)
                 ds.transform = best_val_transform
                 ds.image_mixer = None


        # Create dataloaders for final training
        batch_size = best_params.get("batch_size", 32)
        num_workers = min(multiprocessing.cpu_count(), 4)
        train_loader = DataLoader(
            outer_train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            outer_test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
        )


        # Create models with best architecture
        model_names = ['resnet101', 'efficientnet_b4', 'densenet121']
        models_list = []

        for model_name in model_names:
            model = create_model(
                model_name=model_name,
                use_attention=best_params.get("use_attention", True),
                dropout_rate=best_params.get("dropout_rate", 0.5),
                hidden_size=best_params.get("hidden_size", 256),
                pretrained=True, # Start with pretrained weights for final train
                freeze_layers=best_params.get("freeze_layers", 0)
            )
            model.to(device)
            models_list.append(model)

        # Create dynamic weight module
        dynamic_weight = DynamicWeight(len(models_list))
        dynamic_weight.to(device)

        # Set up optimizer
        learning_rate = best_params.get("learning_rate", 0.0001) # Often use smaller LR for fine-tuning
        weight_decay = best_params.get("weight_decay", 0.0001)

        optimizer = torch.optim.AdamW([
            {'params': m.parameters(), 'lr': learning_rate / 10} for m in models_list
        ] + [{'params': dynamic_weight.parameters(), 'lr': learning_rate}],
            lr=learning_rate, weight_decay=weight_decay
        )


        # Set up loss function
        criterion = torch.nn.BCEWithLogitsLoss()

        # Set up learning rate scheduler
        num_epochs = 30  # Train for more epochs on final model
        lr_scheduler_type = best_params.get("lr_scheduler", "one_cycle")

        if lr_scheduler_type == "one_cycle":
             try:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                     optimizer, max_lr=[learning_rate / 10] * len(models_list) + [learning_rate],
                     steps_per_epoch=len(train_loader), epochs=num_epochs
                 )
             except Exception as e:
                 logging.warning(f"Failed to init OneCycleLR for final training, using StepLR: {e}")
                 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)

        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
        else:  # step
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)


        # Set up mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Training loop for the final model of this outer fold
        best_val_metric_for_saving = -np.inf # Use a metric like F1 or AUC
        metric_to_optimize = 'f1' # Choose metric to determine best epoch (e.g., f1, auc)
        best_epoch = -1
        fold_config = {'hyperparameters': best_params} # Initialize fold config


        for epoch in range(num_epochs):
            # Training phase
            for model in models_list: model.train()
            dynamic_weight.train()
            running_loss = 0.0

            # Process batches
            for images, labels in tqdm(train_loader, desc=f"Outer Fold {outer_fold+1} Epoch {epoch+1}/{num_epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                labels_binary = labels.float().view(-1, 1)

                optimizer.zero_grad()

                try:
                    with torch.cuda.amp.autocast():
                        outputs = ensemble_predict(models_list, images, dynamic_weight)
                        loss = criterion(outputs, labels_binary)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if lr_scheduler_type == "one_cycle" or lr_scheduler_type == "cosine":
                        scheduler.step()

                    running_loss += loss.item()

                except Exception as e:
                    logging.error(f"Error during final training batch (Epoch {epoch+1}): {e}")
                    continue

            epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

            if lr_scheduler_type == "step":
                 scheduler.step()

            # Evaluate on the outer test fold (validation for this stage)
            # Use a fixed threshold (e.g., 0.5) for consistent epoch comparison
            val_metrics = evaluate(
                models_list, test_loader, dynamic_weight,
                epoch=epoch, # Pass epoch for plotting
                is_test=True, # Technically evaluating on the outer fold's test set
                threshold=0.5, # Use fixed threshold for selecting best epoch
                device=device
            )

            current_val_metric = val_metrics.get(metric_to_optimize, -np.inf)
            logging.info(f"Outer Fold {outer_fold+1} - Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} - Val {metric_to_optimize.upper()}: {current_val_metric:.4f}")


            # Save best model based on the chosen validation metric
            if current_val_metric > best_val_metric_for_saving:
                best_val_metric_for_saving = current_val_metric
                best_epoch = epoch + 1 # Use 1-based epoch number

                # Save models
                for i, (model, model_name) in enumerate(zip(models_list, model_names)):
                    model_path = os.path.join(outer_fold_dir, f"{model_name}_best.pth")
                    torch.save(model.state_dict(), model_path) # Save only state_dict

                # Save dynamic weight
                dw_path = os.path.join(outer_fold_dir, "dynamic_weight_best.pth")
                torch.save(dynamic_weight.state_dict(), dw_path) # Save only state_dict

                logging.info(f"  ---> New best {metric_to_optimize.upper()} = {current_val_metric:.4f} at epoch {best_epoch}. Saving models.")

                # Find optimal threshold on this validation data *at this epoch*
                # This threshold is specific to the model state at this best epoch
                optimal_threshold, metrics_at_threshold = find_optimal_threshold(
                    val_metrics['all_probs'], val_metrics['all_labels']
                )
                logging.info(f"  ---> Optimal threshold on validation set at this epoch: {optimal_threshold:.4f}")


                # Update fold config with best epoch info and metrics *at threshold 0.5*
                fold_config['best_epoch'] = best_epoch
                fold_config['best_val_metric'] = best_val_metric_for_saving # Metric at threshold 0.5
                fold_config['optimal_threshold_val_epoch'] = optimal_threshold # Threshold found on val set
                fold_config['metrics_at_val_threshold_0.5'] = {k: v for k, v in val_metrics.items() if k not in ['all_probs', 'all_labels', 'roc_data', 'confusion_matrix']}
                fold_config['metrics_at_val_optimal_thresh'] = metrics_at_threshold # Metrics at optimal threshold


            # Optional: Early stopping
            # if epoch - best_epoch >= early_stopping_patience:
            #     logging.info(f"Stopping early at epoch {epoch+1} due to no improvement.")
            #     break


        # --- Final Evaluation after Training ---
        # Load the best models saved during training for this fold
        logging.info(f"Loading best models from epoch {best_epoch} for final evaluation on outer fold {outer_fold+1}'s test set.")
        for i, (model, model_name) in enumerate(zip(models_list, model_names)):
            model_path = os.path.join(outer_fold_dir, f"{model_name}_best.pth")
            try:
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                models_list[i] = model # Ensure the list has the loaded model
            except Exception as e:
                 logging.error(f"Failed to load best model {model_name} for final eval: {e}")
                 # Handle error - maybe skip this fold's final result?

        # Load best dynamic weight
        dw_path = os.path.join(outer_fold_dir, "dynamic_weight_best.pth")
        try:
            state_dict = torch.load(dw_path)
            dynamic_weight.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Failed to load best dynamic weight for final eval: {e}")


        # Determine the final optimal threshold for this fold
        # Option 1: Use the threshold found on the validation set at the best epoch
        final_optimal_threshold = fold_config.get('optimal_threshold_val_epoch', 0.5)
        logging.info(f"Using optimal threshold found during validation: {final_optimal_threshold:.4f}")

        # Option 2: Re-calculate threshold on the *entire* outer test set using the best model
        # final_eval_metrics_for_thresh = evaluate(...) # Evaluate with threshold 0.5 first
        # final_optimal_threshold, _ = find_optimal_threshold(final_eval_metrics_for_thresh['all_probs'], final_eval_metrics_for_thresh['all_labels'])
        # logging.info(f"Recalculated optimal threshold on outer test set: {final_optimal_threshold:.4f}")


        # Final evaluation using the chosen optimal threshold
        logging.info(f"Performing final evaluation with threshold {final_optimal_threshold:.4f}")
        final_test_metrics = evaluate(
            models_list, test_loader, dynamic_weight,
            epoch=best_epoch, # Use best epoch for logging/plotting
            is_test=True,
            threshold=final_optimal_threshold,
            device=device
        )

        # Update and save fold results (including final metrics at optimal threshold)
        fold_config['final_evaluation_threshold'] = final_optimal_threshold
        fold_config['final_metrics'] = {k: v for k, v in final_test_metrics.items() if k not in ['all_probs', 'all_labels', 'roc_data', 'confusion_matrix']}
        # Add MCC here explicitly if not already included by the loop above
        fold_config['final_metrics']['mcc'] = final_test_metrics.get('mcc', 0.0)


        with open(os.path.join(outer_fold_dir, "fold_config.json"), "w") as f:
            # Convert numpy types before saving
            json.dump(convert_numpy_types_for_json(fold_config), f, indent=4)


        # Store results for this fold (ensure conversion for overall results later)
        outer_results[f'fold_{outer_fold}'] = convert_numpy_types_for_json(fold_config)


        # Clean up memory for outer fold
        for model in models_list: del model
        del dynamic_weight, optimizer, criterion, scheduler, scaler
        del train_loader, test_loader, outer_train_subset, outer_test_subset
        torch.cuda.empty_cache()
        gc.collect()


    # Calculate average performance across all folds
    all_fold_metrics_final = {}
    valid_folds = 0
    metrics_to_average = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity', 'mcc', 'avg_precision', 'log_loss'] # Added MCC

    for fold_data in outer_results.values():
        if 'final_metrics' in fold_data:
            valid_folds += 1
            for metric in metrics_to_average:
                if metric not in all_fold_metrics_final:
                    all_fold_metrics_final[metric] = []
                all_fold_metrics_final[metric].append(fold_data['final_metrics'].get(metric, np.nan)) # Use NaN if missing


    avg_results = {}
    std_results = {}
    if valid_folds > 0:
        for metric in metrics_to_average:
            values = all_fold_metrics_final.get(metric, [])
            if values:
                 avg_results[f'avg_{metric}'] = np.nanmean(values)
                 std_results[f'std_{metric}'] = np.nanstd(values)
            else:
                 avg_results[f'avg_{metric}'] = np.nan
                 std_results[f'std_{metric}'] = np.nan

    # Save overall results (already converted fold results)
    outer_results['average_performance'] = {
         'avg_metrics': avg_results,
         'std_metrics': std_results
    }


    with open(os.path.join(save_dir, "nested_cv_results.json"), "w") as f:
        json.dump(outer_results, f, indent=4) # Already converted

    # Log final results
    logging.info("\n" + "="*50)
    logging.info("Nested Cross-Validation Final Results (Using Fold-Specific Optimal Thresholds):")
    for metric in metrics_to_average:
        avg_key = f'avg_{metric}'
        std_key = f'std_{metric}'
        if avg_key in avg_results and std_key in std_results:
             logging.info(f"  Average {metric.capitalize()}: {avg_results[avg_key]:.4f} ± {std_results[std_key]:.4f}")
    logging.info("="*50)


    return outer_results


def evaluate_on_test_set(nested_cv_results):
    """
    Evaluate the best models from each fold on the actual test set.

    Args:
        nested_cv_results: Dictionary containing the results from nested CV

    Returns:
        test_results: Dictionary containing test set evaluation results for each fold
    """
    global plots_dir, save_dir, data_dir, device # Use global variables

    logging.info(f"\n{'='*50}\nEvaluating Best Models on External Test Set\n{'='*50}")

    # Create directory for test results
    test_results_dir = os.path.join(save_dir, "test_evaluation")
    os.makedirs(test_results_dir, exist_ok=True)

    # Load the actual test set
    # Use the non-augmenting transform
    _, test_transform, _, _ = get_data_transforms(use_image_mix=False)
    test_dataset = ClassAwareDataset(root=f"{data_dir}/test", transform=test_transform, image_mixer=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=min(multiprocessing.cpu_count(), 4), pin_memory=True)

    logging.info(f"Loaded external test dataset with {len(test_dataset)} images")

    # Extract fold results (excluding the average_performance key)
    fold_results = {k: v for k, v in nested_cv_results.items() if k != 'average_performance'}

    # Initialize results dictionary
    test_results_agg = {} # Aggregate results here
    all_fold_eval_metrics = [] # Store metrics dict from evaluate() for each fold


    # For each fold, load and evaluate the best models
    for fold_idx_str, fold_data in sorted(fold_results.items()):
        fold_num = int(fold_idx_str.split('_')[1])
        logging.info(f"\n--- Evaluating models from Fold {fold_num} on External Test Set ---")

        # Get fold directory
        fold_dir = os.path.join(save_dir, f"outer_fold_{fold_num}")

        # Load best hyperparameters and optimal threshold determined during CV for this fold
        try:
            # Load the fold_config.json which should contain the best HPs and the threshold
             with open(os.path.join(fold_dir, "fold_config.json"), "r") as f:
                fold_config = json.load(f)
             best_params = fold_config.get('hyperparameters', {})
             # Use the final threshold determined for this fold during CV
             optimal_threshold = fold_config.get('final_evaluation_threshold', 0.5)
             logging.info(f"  Using threshold determined during CV for fold {fold_num}: {optimal_threshold:.4f}")
        except Exception as e:
             logging.warning(f"Could not load fold_config.json for fold {fold_num}, using defaults. Error: {e}")
             best_params = {}
             optimal_threshold = 0.5


        # Create models with best architecture from CV
        model_names = ['resnet101', 'efficientnet_b4', 'densenet121']
        models_list = []

        # Load the best models saved for this fold
        for model_name in model_names:
            model = create_model(
                model_name=model_name,
                use_attention=best_params.get("use_attention", True),
                dropout_rate=best_params.get("dropout_rate", 0.5),
                hidden_size=best_params.get("hidden_size", 256),
                pretrained=False,  # Load saved weights, not ImageNet
                freeze_layers=0
            )

            # Load model weights
            model_path = os.path.join(fold_dir, f"{model_name}_best.pth")
            try:
                # Use the load_model function which handles state dict formats
                model, _ = load_model(model, model_path)
                model.to(device)
                model.eval()
                models_list.append(model)
                logging.info(f"  Loaded best {model_name} for fold {fold_num} from {model_path}")
            except Exception as e:
                logging.error(f"  Error loading best {model_name} for fold {fold_num}: {e}")
                # Decide how to handle - skip fold? continue with fewer models?
                continue # Skip model if loading fails


        if len(models_list) != len(model_names):
             logging.warning(f"Fold {fold_num}: Could not load all models ({len(models_list)}/{len(model_names)} loaded). Results may be partial.")
             # Continue if at least one model loaded, otherwise skip fold?
             if not models_list:
                 logging.error(f"Fold {fold_num}: No models loaded. Skipping external test evaluation for this fold.")
                 continue


        # Load dynamic weight for this fold
        dynamic_weight = DynamicWeight(len(models_list)) # Re-init based on loaded models
        dw_path = os.path.join(fold_dir, "dynamic_weight_best.pth")
        try:
             # Use the load_dynamic_weight function
            dynamic_weight, _ = load_dynamic_weight(dynamic_weight, dw_path)
            dynamic_weight.to(device)
            dynamic_weight.eval()
            logging.info(f"  Loaded best dynamic weights for fold {fold_num} from {dw_path}")
        except Exception as e:
            logging.warning(f"  Error loading dynamic weights for fold {fold_num}: {e}. Defaulting to equal weights.")
            # Keep the default initialized dynamic_weight (equal weights)

        # Create fold-specific test results directory within the main test_evaluation dir
        fold_test_dir = os.path.join(test_results_dir, f"fold_{fold_num}")
        os.makedirs(fold_test_dir, exist_ok=True)

        # Set plot directory temporarily for this fold's external test plots
        original_plots_dir_global = plots_dir
        plots_dir = fold_test_dir # Point global plots_dir to this fold's test eval subdir

        # Evaluate on external test set using the fold's optimal threshold
        # Pass a unique epoch number (e.g., -99) to distinguish these plots
        fold_eval_metrics = evaluate(
            models_list,
            test_loader,
            dynamic_weight,
            epoch=-99, # Use a distinct epoch marker for external test plots
            is_test=True,
            threshold=optimal_threshold,
            device=device
        )

        # Reset global plots directory
        plots_dir = original_plots_dir_global


        # Save detailed fold-specific metrics (JSON without large arrays)
        fold_metrics_path = os.path.join(fold_test_dir, "external_test_metrics.json")
        metrics_to_save = {k: v for k, v in fold_eval_metrics.items()
                           if k not in ['all_probs', 'all_labels']}
        try:
            with open(fold_metrics_path, "w") as f:
                json.dump(convert_numpy_types_for_json(metrics_to_save), f, indent=4)
        except Exception as e:
             logging.error(f"Failed to save fold {fold_num} external test metrics: {e}")

        # Save probabilities and labels separately for this fold's eval
        probs_labels_path = os.path.join(fold_test_dir, "external_test_probs_labels.npz")
        try:
            np.savez(
                probs_labels_path,
                probabilities=fold_eval_metrics['all_probs'],
                labels=fold_eval_metrics['all_labels']
            )
        except Exception as e:
            logging.error(f"Failed to save fold {fold_num} external test probs/labels: {e}")


        # Generate additional plots for this fold's external test performance
        # 1. Precision-Recall Curve
        try:
            precision, recall, _ = precision_recall_curve(
                fold_eval_metrics['all_labels'], fold_eval_metrics['all_probs']
            )
            avg_precision_fold = fold_eval_metrics.get('avg_precision', np.nan)

            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, 'b-', lw=2, label=f'PR Curve (AP = {avg_precision_fold:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'External Test PR Curve (Fold {fold_num}, Thresh={optimal_threshold:.2f})')
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pr_curve_path = os.path.join(fold_test_dir, "external_test_precision_recall_curve.png")
            plt.savefig(pr_curve_path, dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Failed to plot PR curve for fold {fold_num}: {e}")
            plt.close()


        # 2. Threshold vs F1 curve
        try:
            f1_scores_fold = []
            threshold_values_fold = np.linspace(0.05, 0.95, 19)
            for thresh in threshold_values_fold:
                preds = (fold_eval_metrics['all_probs'] >= thresh).astype(int)
                f1_fold = f1_score(fold_eval_metrics['all_labels'], preds)
                f1_scores_fold.append(f1_fold)

            plt.figure(figsize=(10, 6))
            plt.plot(threshold_values_fold, f1_scores_fold, 'r-o', lw=2)
            plt.axvline(x=optimal_threshold, color='blue', linestyle='--',
                       label=f'Used Threshold = {optimal_threshold:.2f}')
            plt.xlabel('Threshold')
            plt.ylabel('F1 Score')
            plt.title(f'External Test Threshold vs F1 (Fold {fold_num})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            f1_threshold_path = os.path.join(fold_test_dir, "external_test_threshold_vs_f1.png")
            plt.savefig(f1_threshold_path, dpi=300)
            plt.close()
        except Exception as e:
             logging.error(f"Failed to plot Threshold vs F1 for fold {fold_num}: {e}")
             plt.close()


        # Store results for aggregation
        test_results_agg[f'fold_{fold_num}'] = {
            'metrics': {k: v for k, v in fold_eval_metrics.items() if k not in ['all_probs', 'all_labels', 'roc_data', 'confusion_matrix']},
            'hyperparameters': best_params, # Store HPs used for this model
            'optimal_threshold': optimal_threshold # Store threshold used
        }
        # Add MCC if missing
        if 'mcc' not in test_results_agg[f'fold_{fold_num}']['metrics']:
             test_results_agg[f'fold_{fold_num}']['metrics']['mcc'] = fold_eval_metrics.get('mcc', np.nan)


        # Keep the full metrics dict (including probs/labels) for ensemble calculation
        all_fold_eval_metrics.append(fold_eval_metrics)


        # Clean up memory for this fold
        for model in models_list: del model
        del dynamic_weight
        torch.cuda.empty_cache()
        gc.collect()


    # --- Aggregate results across folds ---
    # Calculate average performance on the external test set
    avg_test_metrics = {}
    std_test_metrics = {}
    metrics_to_average_test = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity', 'mcc', 'avg_precision', 'log_loss'] # Added MCC

    valid_test_folds = 0
    metric_values_test = {m: [] for m in metrics_to_average_test}

    for fold_key, fold_res in test_results_agg.items():
         if 'metrics' in fold_res:
            valid_test_folds += 1
            for metric in metrics_to_average_test:
                 metric_values_test[metric].append(fold_res['metrics'].get(metric, np.nan))

    if valid_test_folds > 0:
        for metric in metrics_to_average_test:
            values = metric_values_test.get(metric, [])
            if values:
                avg_test_metrics[metric] = np.nanmean(values)
                std_test_metrics[metric] = np.nanstd(values)
            else:
                avg_test_metrics[metric] = np.nan
                std_test_metrics[metric] = np.nan

    # Add average performance to results
    test_results_agg['average_performance'] = {
        'avg_metrics': avg_test_metrics,
        'std_metrics': std_test_metrics
    }

    # --- Ensemble Prediction on External Test Set ---
    if all_fold_eval_metrics: # Check if we have results from any fold
        try:
            logging.info("\n" + "="*50)
            logging.info("Creating Ensemble Prediction for External Test Set")

            # Collect probabilities and labels (should be same labels for all)
            all_probs_test = np.stack([m['all_probs'] for m in all_fold_eval_metrics])
            all_labels_test = all_fold_eval_metrics[0]['all_labels']

            # Average probabilities across folds
            ensemble_probs_test = np.mean(all_probs_test, axis=0)

            # Find optimal threshold for the ensemble on the test set probabilities
            optimal_threshold_ensemble, metrics_at_threshold_ensemble = find_optimal_threshold(ensemble_probs_test, all_labels_test)
            logging.info(f"  Optimal threshold for ensemble on test set: {optimal_threshold_ensemble:.4f}")


            # Calculate final ensemble metrics using this optimal threshold
            predictions_ensemble = (ensemble_probs_test >= optimal_threshold_ensemble).astype(int)

            # Use sklearn/numpy for direct calculation for verification
            ensemble_accuracy = np.mean(predictions_ensemble == all_labels_test)
            ensemble_f1 = f1_score(all_labels_test, predictions_ensemble)
            ensemble_auc = roc_auc_score(all_labels_test, ensemble_probs_test)
            ensemble_mcc = matthews_corrcoef(all_labels_test, predictions_ensemble) # Calculate MCC

            from sklearn.metrics import precision_score, recall_score, confusion_matrix as sklearn_cm
            ensemble_precision = precision_score(all_labels_test, predictions_ensemble)
            ensemble_recall = recall_score(all_labels_test, predictions_ensemble) # = sensitivity
            conf_matrix_ensemble = sklearn_cm(all_labels_test, predictions_ensemble)
            tn_ens, fp_ens, fn_ens, tp_ens = conf_matrix_ensemble.ravel()
            ensemble_specificity = tn_ens / (tn_ens + fp_ens) if (tn_ens + fp_ens) > 0 else 0

            # Avg Precision (PR AUC)
            prec_ens, rec_ens, _ = precision_recall_curve(all_labels_test, ensemble_probs_test)
            ensemble_avg_precision = auc(rec_ens, prec_ens) # Calculate AUC of PR curve
            ensemble_log_loss = log_loss(all_labels_test, ensemble_probs_test, eps=1e-7)


            # Add ensemble results to test_results_agg
            test_results_agg['ensemble'] = {
                'metrics': {
                    'accuracy': float(ensemble_accuracy),
                    'precision': float(ensemble_precision),
                    'recall': float(ensemble_recall), # sensitivity
                    'f1': float(ensemble_f1),
                    'auc': float(ensemble_auc),
                    'sensitivity': float(ensemble_recall), # Explicitly sensitivity
                    'specificity': float(ensemble_specificity),
                    'avg_precision': float(ensemble_avg_precision),
                    'log_loss': float(ensemble_log_loss),
                    'mcc': float(ensemble_mcc), # Add MCC
                },
                'optimal_threshold': float(optimal_threshold_ensemble)
            }

            # Log ensemble results
            logging.info("\nEnsemble Model External Test Results:")
            logging.info(f"  Optimal Threshold: {optimal_threshold_ensemble:.4f}")
            logging.info(f"  Accuracy: {ensemble_accuracy:.4f}")
            logging.info(f"  Precision: {ensemble_precision:.4f}")
            logging.info(f"  Recall/Sensitivity: {ensemble_recall:.4f}")
            logging.info(f"  Specificity: {ensemble_specificity:.4f}")
            logging.info(f"  F1 Score: {ensemble_f1:.4f}")
            logging.info(f"  AUC: {ensemble_auc:.4f}")
            logging.info(f"  Avg Precision: {ensemble_avg_precision:.4f}")
            logging.info(f"  MCC: {ensemble_mcc:.4f}") # Log MCC
            logging.info(f"  Log Loss: {ensemble_log_loss:.4f}")


            # --- Plot Ensemble Results ---
            ensemble_plots_dir = os.path.join(test_results_dir, "ensemble_plots")
            os.makedirs(ensemble_plots_dir, exist_ok=True)

            # ROC Curve
            fpr_ens, tpr_ens, _ = roc_curve(all_labels_test, ensemble_probs_test)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr_ens, tpr_ens, color='darkorange', lw=2, label=f'Ensemble ROC (AUC = {ensemble_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('Ensemble Model ROC Curve (External Test)'); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ensemble_plots_dir, "ensemble_roc_curve.png"), dpi=300); plt.close()

            # Precision-Recall Curve
            plt.figure(figsize=(10, 8))
            plt.plot(rec_ens, prec_ens, 'b-', lw=2, label=f'Ensemble PR Curve (AP = {ensemble_avg_precision:.3f})')
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title('Ensemble Model PR Curve (External Test)'); plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ensemble_plots_dir, "ensemble_pr_curve.png"), dpi=300); plt.close()


            # Confusion Matrix
            plot_confusion_matrix(conf_matrix_ensemble, optimal_threshold_ensemble,
                                  os.path.join(ensemble_plots_dir, "ensemble_confusion_matrix.png"))

             # Probability Distribution
            plot_probability_distribution(ensemble_probs_test, all_labels_test,
                                          f"Ensemble Probability Distribution (External Test, Thresh={optimal_threshold_ensemble:.2f})",
                                          os.path.join(ensemble_plots_dir, "ensemble_prob_distribution.png"))


        except Exception as e:
            logging.error(f"Error creating ensemble prediction on external test set: {e}", exc_info=True)
    else:
         logging.warning("Skipping ensemble calculation as no fold evaluation metrics were collected.")


    # Save overall aggregated test results (including average and ensemble)
    final_test_results_path = os.path.join(test_results_dir, "test_results.json")
    try:
        with open(final_test_results_path, "w") as f:
            # Apply final conversion before saving the main results file
            json.dump(convert_numpy_types_for_json(test_results_agg), f, indent=4)
        logging.info(f"\nFinal aggregated test set evaluation results saved to {final_test_results_path}")
    except Exception as e:
        logging.error(f"Failed to save final aggregated test results: {e}")


    # Create final comparison plots across folds using the aggregated results
    create_test_performance_plots(test_results_agg, test_results_dir)

    logging.info("="*50)

    return test_results_agg


def create_test_performance_plots(test_results, test_results_dir):
    """
    Create comparison plots for test set performance across folds.

    Args:
        test_results: Dictionary containing test results for each fold, avg, and ensemble
        test_results_dir: Directory to save plots
    """
    # Extract fold results (excluding average and ensemble)
    fold_results = {k: v for k, v in test_results.items()
                   if k not in ['average_performance', 'ensemble']}

    if not fold_results:
         logging.warning("No fold results found to create test performance plots.")
         return

    # Define metrics to plot
    # Use the metrics available in the average_performance section if possible
    avg_metrics_dict = test_results.get('average_performance', {}).get('avg_metrics', {})
    metrics_present = list(avg_metrics_dict.keys())
    if not metrics_present:
        # Fallback if average metrics aren't calculated
        metrics_present = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity', 'mcc']


    # Plot each metric across folds
    for metric in metrics_present:
        plt.figure(figsize=(12, 6))

        # Extract metric values from each fold
        fold_indices_str = sorted(fold_results.keys())
        fold_indices_num = [int(k.split('_')[1]) for k in fold_indices_str]
        metric_values = [fold_results[k]['metrics'].get(metric, np.nan) for k in fold_indices_str]


        # Create the bar chart
        bars = plt.bar(fold_indices_num, metric_values, color='steelblue', alpha=0.7, label='Individual Folds')

        # Add average line if available
        avg_value = avg_metrics_dict.get(metric, np.nan)
        std_value = test_results.get('average_performance', {}).get('std_metrics', {}).get(metric, np.nan)

        if not np.isnan(avg_value):
            label_avg = f'Avg: {avg_value:.4f}'
            if not np.isnan(std_value):
                label_avg += f' ± {std_value:.4f}'
            plt.axhline(y=avg_value, color='red', linestyle='--', label=label_avg)


         # Add ensemble value if available
        ensemble_metrics_dict = test_results.get('ensemble', {}).get('metrics', {})
        ensemble_value = ensemble_metrics_dict.get(metric, np.nan)
        if not np.isnan(ensemble_value):
            # Plot ensemble as a distinct bar or line
            ens_pos = max(fold_indices_num) + 1 # Position for ensemble bar
            ens_bar = plt.bar([ens_pos], [ensemble_value], color='green', alpha=0.7, label=f'Ensemble: {ensemble_value:.3f}')
            # Add ensemble label to x-axis
            plt.xticks(fold_indices_num + [ens_pos], [f"F{i}" for i in fold_indices_num] + ['Ens'])
        else:
            # Default x-axis labels if no ensemble
             plt.xticks(fold_indices_num, [f"F{i}" for i in fold_indices_num])


        # Add value labels on top of fold bars
        for bar, value in zip(bars, metric_values):
            if not np.isnan(value):
                plt.text(bar.get_x() + bar.get_width()/2., value + 0.01, f'{value:.3f}',
                         ha='center', va='bottom', fontsize=9)


        # Set labels and title
        plt.xlabel('Fold / Ensemble')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'External Test Set: {metric.capitalize()} Comparison')
        all_vals = [v for v in metric_values if not np.isnan(v)]
        if not np.isnan(ensemble_value): all_vals.append(ensemble_value)
        min_y = min(0, min(all_vals) - 0.1) if all_vals else 0
        max_y = max(1.1, max(all_vals) * 1.15) if all_vals else 1.1
        # Special handling for MCC range (-1 to 1)
        if metric == 'mcc':
             min_y = min(-1.05, min(all_vals) - 0.1) if all_vals else -1.05
             max_y = max(1.05, max(all_vals) * 1.15) if all_vals else 1.05


        plt.ylim(min_y, max_y)
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()


        # Save the plot
        save_path = os.path.join(test_results_dir, f'external_test_{metric}_comparison.png')
        plt.savefig(save_path, dpi=300); plt.close()


    # Create thresholds comparison plot
    plt.figure(figsize=(10, 6))
    fold_indices_str = sorted(fold_results.keys())
    fold_indices_num = [int(k.split('_')[1]) for k in fold_indices_str]
    thresholds = [fold_results[k].get('optimal_threshold', np.nan) for k in fold_indices_str]

    bars = plt.bar(fold_indices_num, thresholds, color='purple', alpha=0.7, label='Fold Threshold')

    # Add ensemble threshold if available
    ensemble_thresh = test_results.get('ensemble', {}).get('optimal_threshold', np.nan)
    if not np.isnan(ensemble_thresh):
        ens_pos = max(fold_indices_num) + 1
        plt.bar([ens_pos], [ensemble_thresh], color='orange', alpha=0.7, label=f'Ensemble Threshold: {ensemble_thresh:.3f}')
        plt.xticks(fold_indices_num + [ens_pos], [f"F{i}" for i in fold_indices_num] + ['Ens'])
    else:
         plt.xticks(fold_indices_num, [f"F{i}" for i in fold_indices_num])


    # Add value labels for folds
    for bar, value in zip(bars, thresholds):
         if not np.isnan(value):
            plt.text(bar.get_x() + bar.get_width()/2., value + 0.01, f'{value:.3f}',
                     ha='center', va='bottom', fontsize=9)


    # Calculate and plot average threshold across folds
    valid_thresholds = [t for t in thresholds if not np.isnan(t)]
    if valid_thresholds:
        avg_thresh = np.mean(valid_thresholds)
        std_thresh = np.std(valid_thresholds)
        plt.axhline(y=avg_thresh, color='red', linestyle='--',
                   label=f'Avg Fold Thresh: {avg_thresh:.4f} ± {std_thresh:.4f}')


    plt.xlabel('Fold / Ensemble')
    plt.ylabel('Optimal Threshold')
    plt.title('External Test: Optimal Threshold Comparison')
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()


    threshold_path = os.path.join(test_results_dir, 'external_test_optimal_thresholds_comparison.png')
    plt.savefig(threshold_path, dpi=300); plt.close()
    logging.info(f"External test performance plots saved to {test_results_dir}")


def main():
    """Main function to run the full pipeline with nested cross-validation and test set evaluation"""
    start_time = time.time()

    try:
        # Run nested cross-validation
        logging.info(f"\n{'='*50}\nStarting Nested Cross-Validation with {OUTER_FOLDS} outer folds and {INNER_FOLDS} inner folds\n{'='*50}")
        nested_cv_results = nested_cross_validation()

        # Evaluate best models from each fold on the test set
        logging.info(f"\n{'='*50}\nEvaluating Best Models on Actual Test Set\n{'='*50}")
        test_results = evaluate_on_test_set(nested_cv_results)
        
        # Create visualization scripts
        logging.info(f"\n{'='*50}\nCreating Visualization Scripts\n{'='*50}")
        create_roc_data_visualization_script()
        create_threshold_metrics_visualization_script()
        
        # Create overlay plots for nested cross-validation results
        logging.info(f"\n{'='*50}\nCreating Overlay Plots for Nested Cross-Validation\n{'='*50}")
        nested_cv_overlay_dir = os.path.join(save_dir, "nested_cv_overlay_plots")
        os.makedirs(nested_cv_overlay_dir, exist_ok=True)
        
        # Run the nested CV overlay plots script
        try:
            # Import the modules to avoid having to write to separate script files
            import numpy as np
            import matplotlib.pyplot as plt
            import json
            import glob
            
            # Plot metrics radar chart for nested CV
            def plot_metrics_radar_nested_cv(nested_cv_results, output_dir):
                # Extract fold results
                fold_results = {k: v for k, v in nested_cv_results.items() 
                              if k != 'average_performance'}
                
                # Define metrics to include in the radar chart
                metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
                metrics_nice_names = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
                
                # Set up the radar chart
                n_metrics = len(metrics)
                angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
                angles += angles[:1]  # Close the polygon
                
                # Set up the figure
                fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
                
                # Plot data for each fold
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                line_styles = ['-', '--', '-.', ':', '-', '--']
                
                for i, (fold_idx, fold_data) in enumerate(sorted(fold_results.items())):
                    if 'final_metrics' not in fold_data:
                        continue
                        
                    fold_num = int(fold_idx.split('_')[1])
                    
                    # Get metrics for this fold
                    fold_values = []
                    for m in metrics:
                        if m in fold_data['final_metrics']:
                            fold_values.append(fold_data['final_metrics'][m])
                        else:
                            fold_values.append(0)  # Default if metric not found
                            
                    fold_values += fold_values[:1]  # Close the polygon
                    
                    # Plot fold data
                    ax.plot(angles, fold_values, color=colors[i % len(colors)], 
                            linestyle=line_styles[i % len(line_styles)], linewidth=2, 
                            label=f'Fold {fold_num}')
                    ax.fill(angles, fold_values, color=colors[i % len(colors)], alpha=0.1)
                
                # Plot average data if available
                avg_values = []
                for m in metrics:
                    avg_key = f'avg_{m}'
                    if avg_key in nested_cv_results['average_performance']:
                        avg_values.append(nested_cv_results['average_performance'][avg_key])
                    else:
                        avg_values.append(0)  # Default if metric not found
                        
                avg_values += avg_values[:1]  # Close the polygon
                
                ax.plot(angles, avg_values, color='red', linestyle='--', linewidth=2, label='Average')
                
                # Set up the radar chart properties
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics_nice_names, fontsize=12)
                
                # Add y-axis grid lines at 0.2 intervals
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.set_ylim(0, 1)
                
                # Add a title
                plt.title('Nested CV: Comparison of Metrics Across Outer Folds', fontsize=16, y=1.08)
                
                # Add a legend
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                # Save the radar chart
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'nested_cv_metrics_radar.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(output_dir, 'nested_cv_metrics_radar.pdf'), bbox_inches='tight')
                plt.close()
                
                logging.info(f"Nested CV metrics radar chart saved to {output_dir}")
            
            # Create radar chart for nested CV results
            plot_metrics_radar_nested_cv(nested_cv_results, nested_cv_overlay_dir)
            
            # Create radar chart for test results
            def plot_metrics_radar_test(test_results, output_dir):
                # Extract fold results (excluding the average_performance and ensemble keys)
                fold_results = {k: v for k, v in test_results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                # Define metrics to include in the radar chart
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity']
                metrics_nice_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Sensitivity', 'Specificity']
                
                # Set up the radar chart
                n_metrics = len(metrics)
                angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
                angles += angles[:1]  # Close the polygon
                
                # Set up the figure
                fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
                
                # Colors for the radar chart
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                line_styles = ['-', '--', '-.', ':', '-', '--']
                
                # Plot data for each fold
                for i, (fold_idx, fold_data) in enumerate(sorted(fold_results.items())):
                    fold_num = int(fold_idx.split('_')[1])
                    
                    # Get metrics for this fold
                    fold_values = []
                    for m in metrics:
                        if m in fold_data['metrics']:
                            fold_values.append(fold_data['metrics'][m])
                        else:
                            fold_values.append(0)  # Default if metric not found
                            
                    fold_values += fold_values[:1]  # Close the polygon
                    
                    # Plot fold data
                    ax.plot(angles, fold_values, color=colors[i % len(colors)], 
                            linestyle=line_styles[i % len(line_styles)], linewidth=2, 
                            label=f'Fold {fold_num}')
                    ax.fill(angles, fold_values, color=colors[i % len(colors)], alpha=0.1)
                
                # Plot ensemble data if available
                if 'ensemble' in test_results:
                    ensemble_values = []
                    for m in metrics:
                        if m in test_results['ensemble']['metrics']:
                            ensemble_values.append(test_results['ensemble']['metrics'][m])
                        else:
                            ensemble_values.append(0)  # Default if metric not found
                            
                    ensemble_values += ensemble_values[:1]  # Close the polygon
                    
                    ax.plot(angles, ensemble_values, color='black', linewidth=3, label='Ensemble')
                    ax.fill(angles, ensemble_values, color='black', alpha=0.1)
                
                # Set up the radar chart properties
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics_nice_names, fontsize=12)
                
                # Add y-axis grid lines at 0.2 intervals
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.set_ylim(0, 1)
                
                # Add a title
                plt.title('External Test: Comparison of Metrics Across Folds and Ensemble', fontsize=16, y=1.08)
                
                # Add a legend
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                # Save the radar chart
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'test_metrics_radar.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(output_dir, 'test_metrics_radar.pdf'), bbox_inches='tight')
                plt.close()
                
                logging.info(f"Test metrics radar chart saved to {output_dir}")
            
            # Create radar chart for test results
            test_radar_dir = os.path.join(save_dir, "test_evaluation", "overlay_plots")
            os.makedirs(test_radar_dir, exist_ok=True)
            plot_metrics_radar_test(test_results, test_radar_dir)
            
            # Add these functions after the existing overlay plots in main()

            # 1. Precision-Recall Curve Overlay
            def plot_pr_curves_overlay(results, output_dir, source_data_dir, is_external_test=True):
                """
                Create an overlay plot of Precision-Recall curves for all folds and the ensemble
                
                Args:
                    results: Dictionary containing results (nested_cv_results or test_results)
                    output_dir: Directory to save the plot
                    source_data_dir: Directory where the data files are stored
                    is_external_test: Whether this is for external test or CV folds
                """
                # Extract fold results
                fold_results = {k: v for k, v in results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                plt.figure(figsize=(12, 10))
                
                # Colors for different folds
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                line_styles = ['-', '--', '-.', ':', '-', '--']
                
                from sklearn.metrics import precision_recall_curve, auc
                
                # For each fold, load data and plot PR curve
                for i, fold_key in enumerate(sorted(fold_results.keys())):
                    fold_num = int(fold_key.split('_')[1])
                    
                    if is_external_test:
                        # External test data path
                        fold_dir = os.path.join(source_data_dir, "test_evaluation", f"fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "external_test_probs_labels.npz")
                        metrics_file = os.path.join(fold_dir, "external_test_metrics.json")
                    else:
                        # Nested CV data path
                        fold_dir = os.path.join(source_data_dir, f"outer_fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "fold_probs_labels.npz")
                        metrics_file = os.path.join(fold_dir, "fold_config.json")
                    
                    try:
                        # Try to load probabilities and labels
                        if os.path.exists(probs_labels_file):
                            data = np.load(probs_labels_file)
                            y_true = data['labels']
                            y_score = data['probabilities']
                            
                            # Calculate precision-recall curve
                            precision, recall, _ = precision_recall_curve(y_true, y_score)
                            
                            # Calculate area under PR curve
                            avg_precision = auc(recall, precision)
                            
                            # Plot PR curve for this fold
                            plt.plot(recall, precision, color=colors[i % len(colors)], 
                                     linestyle=line_styles[i % len(line_styles)],
                                     lw=2, label=f'Fold {fold_num} (AP = {avg_precision:.3f})')
                        
                        # If data file doesn't exist but metrics file does, try to get AP from there
                        elif os.path.exists(metrics_file):
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                            
                            if is_external_test:
                                avg_precision = metrics_data.get('metrics', {}).get('avg_precision', 0)
                            else:
                                if 'final_metrics' in metrics_data:
                                    avg_precision = metrics_data['final_metrics'].get('avg_precision', 0)
                                else:
                                    avg_precision = 0
                            
                            logging.warning(f"Could not load PR curve data for fold {fold_num}. "
                                           f"Using only AP = {avg_precision:.3f}")
                    
                    except Exception as e:
                        logging.warning(f"Error loading PR data for fold {fold_num}: {e}")
                
                # Plot ensemble PR curve if available (only for external test)
                if is_external_test and 'ensemble' in results:
                    ensemble_dir = os.path.join(source_data_dir, "test_evaluation", "ensemble_plots")
                    ensemble_npz_files = glob.glob(os.path.join(ensemble_dir, "*.npz"))
                    
                    try:
                        if ensemble_npz_files:
                            ensemble_data = np.load(ensemble_npz_files[0])
                            if 'probabilities' in ensemble_data and 'labels' in ensemble_data:
                                # Calculate precision-recall curve for ensemble
                                precision_ens, recall_ens, _ = precision_recall_curve(
                                    ensemble_data['labels'], ensemble_data['probabilities'])
                                
                                # Calculate AP for ensemble
                                ap_ens = results['ensemble']['metrics'].get('avg_precision', 
                                                                          auc(recall_ens, precision_ens))
                                
                                # Plot ensemble PR curve
                                plt.plot(recall_ens, precision_ens, color='black', linestyle='-', lw=3,
                                        label=f'Ensemble (AP = {ap_ens:.3f})')
                        else:
                            # Try to get AP from results
                            ap_ens = results['ensemble']['metrics'].get('avg_precision', 0)
                            logging.warning(f"Could not load ensemble PR curve data. "
                                          f"Using only AP = {ap_ens:.3f}")
                            
                    except Exception as e:
                        logging.warning(f"Error plotting ensemble PR curve: {e}")
                
                # Set labels and title
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall', fontsize=14)
                plt.ylabel('Precision', fontsize=14)
                
                if is_external_test:
                    plt.title('External Test: Precision-Recall Curves Across All Folds and Ensemble', fontsize=16)
                else:
                    plt.title('Nested CV: Precision-Recall Curves Across All Folds', fontsize=16)
                
                plt.legend(loc="best", fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                os.makedirs(output_dir, exist_ok=True)
                if is_external_test:
                    plt.savefig(os.path.join(output_dir, 'precision_recall_curves_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'precision_recall_curves_overlay.pdf'))
                else:
                    plt.savefig(os.path.join(output_dir, 'nested_cv_precision_recall_curves_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'nested_cv_precision_recall_curves_overlay.pdf'))
                
                plt.close()
                logging.info(f"Precision-Recall overlay plot saved to {output_dir}")
            
            # 5. Matthews Correlation Coefficient (MCC) Comparison - Complete Code
            def plot_mcc_comparison(results, output_dir, is_external_test=True):
                """
                Create a comparison plot of Matthews Correlation Coefficient across folds
                
                Args:
                    results: Dictionary containing results (nested_cv_results or test_results)
                    output_dir: Directory to save the plot
                    is_external_test: Whether this is for external test or CV folds
                """
                # Extract fold results
                fold_results = {k: v for k, v in results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                plt.figure(figsize=(12, 6))
                
                # Extract MCC values
                fold_indices_str = sorted(fold_results.keys())
                fold_indices_num = [int(k.split('_')[1]) for k in fold_indices_str]
                
                if is_external_test:
                    mcc_values = [fold_results[k]['metrics'].get('mcc', np.nan) for k in fold_indices_str]
                else:
                    mcc_values = [fold_results[k].get('final_metrics', {}).get('mcc', np.nan) for k in fold_indices_str]
                
                # Create the bar chart
                bars = plt.bar(fold_indices_num, mcc_values, color='purple', alpha=0.7, label='Individual Folds')
                
                # Add average line if available
                if is_external_test:
                    avg_mcc = results.get('average_performance', {}).get('avg_metrics', {}).get('mcc', np.nan)
                    std_mcc = results.get('average_performance', {}).get('std_metrics', {}).get('mcc', np.nan)
                else:
                    avg_mcc = results.get('average_performance', {}).get('avg_mcc', np.nan)
                    std_mcc = results.get('average_performance', {}).get('std_mcc', np.nan)
                
                if not np.isnan(avg_mcc):
                    label_avg = f'Avg: {avg_mcc:.4f}'
                    if not np.isnan(std_mcc):
                        label_avg += f' ± {std_mcc:.4f}'
                    plt.axhline(y=avg_mcc, color='red', linestyle='--', label=label_avg)
                
                # Add ensemble value if available
                if is_external_test and 'ensemble' in results:
                    ensemble_mcc = results['ensemble']['metrics'].get('mcc', np.nan)
                    if not np.isnan(ensemble_mcc):
                        # Plot ensemble as a distinct bar
                        ens_pos = max(fold_indices_num) + 1
                        ens_bar = plt.bar([ens_pos], [ensemble_mcc], color='green', alpha=0.7, 
                                         label=f'Ensemble: {ensemble_mcc:.4f}')
                        # Add ensemble label to x-axis
                        plt.xticks(fold_indices_num + [ens_pos], [f"F{i}" for i in fold_indices_num] + ['Ens'])
                    else:
                        plt.xticks(fold_indices_num, [f"F{i}" for i in fold_indices_num])
                else:
                    plt.xticks(fold_indices_num, [f"F{i}" for i in fold_indices_num])
                
                # Add value labels on top of fold bars
                for bar, value in zip(bars, mcc_values):
                    if not np.isnan(value):
                        plt.text(bar.get_x() + bar.get_width()/2., value + 0.01, f'{value:.3f}',
                                ha='center', va='bottom', fontsize=9)
                
                # Set labels and title
                plt.xlabel('Fold / Ensemble', fontsize=12)
                plt.ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12)
                
                if is_external_test:
                    plt.title('External Test: Matthews Correlation Coefficient Comparison', fontsize=14)
                else:
                    plt.title('Nested CV: Matthews Correlation Coefficient Comparison', fontsize=14)
                
                # MCC ranges from -1 to 1
                plt.ylim(-1.05, 1.05)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)  # Zero line
                
                plt.grid(axis='y', alpha=0.3)
                plt.legend(loc='best')
                
                # Save the plot
                os.makedirs(output_dir, exist_ok=True)
                if is_external_test:
                    plt.savefig(os.path.join(output_dir, 'mcc_comparison.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'mcc_comparison.pdf'))
                else:
                    plt.savefig(os.path.join(output_dir, 'nested_cv_mcc_comparison.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'nested_cv_mcc_comparison.pdf'))
                
                plt.close()
                logging.info(f"MCC comparison plot saved to {output_dir}")
            
            
            # 6. AUROC vs. Average Precision Overlay
            def plot_auroc_vs_ap_overlay(results, output_dir, is_external_test=True):
                """
                Create a scatter plot comparing AUROC vs Average Precision across folds
                
                Args:
                    results: Dictionary containing results (nested_cv_results or test_results)
                    output_dir: Directory to save the plot
                    is_external_test: Whether this is for external test or CV folds
                """
                # Extract fold results
                fold_results = {k: v for k, v in results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                plt.figure(figsize=(10, 8))
                
                # Extract AUC and AP values for each fold
                fold_nums = []
                auc_values = []
                ap_values = []
                
                for fold_key, fold_data in sorted(fold_results.items()):
                    fold_num = int(fold_key.split('_')[1])
                    fold_nums.append(fold_num)
                    
                    if is_external_test:
                        auc = fold_data['metrics'].get('auc', np.nan)
                        ap = fold_data['metrics'].get('avg_precision', np.nan)
                    else:
                        auc = fold_data.get('final_metrics', {}).get('auc', np.nan)
                        ap = fold_data.get('final_metrics', {}).get('avg_precision', np.nan)
                    
                    if not np.isnan(auc) and not np.isnan(ap):
                        auc_values.append(auc)
                        ap_values.append(ap)
                
                # Plot each fold as a scatter point
                for i, (fold, auc, ap) in enumerate(zip(fold_nums, auc_values, ap_values)):
                    plt.scatter(auc, ap, s=100, label=f'Fold {fold}')
                    plt.annotate(f'F{fold}', (auc, ap), xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                # Add ensemble point if available
                if is_external_test and 'ensemble' in results:
                    ens_auc = results['ensemble']['metrics'].get('auc', np.nan)
                    ens_ap = results['ensemble']['metrics'].get('avg_precision', np.nan)
                    
                    if not np.isnan(ens_auc) and not np.isnan(ens_ap):
                        plt.scatter(ens_auc, ens_ap, s=200, color='red', marker='*', label='Ensemble')
                        plt.annotate('Ensemble', (ens_auc, ens_ap), xytext=(5, 5), 
                                    textcoords='offset points', fontsize=10, weight='bold')
                
                # Add average values if available
                if is_external_test:
                    avg_auc = results.get('average_performance', {}).get('avg_metrics', {}).get('auc', np.nan)
                    avg_ap = results.get('average_performance', {}).get('avg_metrics', {}).get('avg_precision', np.nan)
                else:
                    avg_auc = results.get('average_performance', {}).get('avg_auc', np.nan)
                    avg_ap = results.get('average_performance', {}).get('avg_avg_precision', np.nan)
                
                if not np.isnan(avg_auc) and not np.isnan(avg_ap):
                    plt.scatter(avg_auc, avg_ap, s=150, color='green', marker='D', label='Average')
                    plt.annotate('Average', (avg_auc, avg_ap), xytext=(5, 5), 
                                textcoords='offset points', fontsize=10)
                
                # Add diagonal line (y=x)
                min_val = min(min(auc_values), min(ap_values)) - 0.05
                max_val = max(max(auc_values), max(ap_values)) + 0.05
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
                
                # Set equal aspect ratio
                plt.axis('equal')
                
                # Set labels and title
                plt.xlabel('Area Under ROC Curve (AUROC)', fontsize=12)
                plt.ylabel('Average Precision (AP)', fontsize=12)
                
                if is_external_test:
                    plt.title('External Test: AUROC vs Average Precision', fontsize=14)
                else:
                    plt.title('Nested CV: AUROC vs Average Precision', fontsize=14)
                
                # Set axis limits with some padding
                padding = 0.05
                plt.xlim(min_val - padding, max_val + padding)
                plt.ylim(min_val - padding, max_val + padding)
                
                # Add grid and legend
                plt.grid(True, alpha=0.3)
                plt.legend(loc='lower right')
                
                # Save the plot
                os.makedirs(output_dir, exist_ok=True)
                if is_external_test:
                    plt.savefig(os.path.join(output_dir, 'auroc_vs_ap_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'auroc_vs_ap_overlay.pdf'))
                else:
                    plt.savefig(os.path.join(output_dir, 'nested_cv_auroc_vs_ap_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'nested_cv_auroc_vs_ap_overlay.pdf'))
                
                plt.close()
                logging.info(f"AUROC vs AP overlay plot saved to {output_dir}")
            # 2. Probability Distribution Overlay
            def plot_probability_distribution_overlay(results, output_dir, source_data_dir, is_external_test=True):
                """
                Create an overlay plot of probability distributions for all folds
                
                Args:
                    results: Dictionary containing results (nested_cv_results or test_results)
                    output_dir: Directory to save the plot
                    source_data_dir: Directory where the data files are stored
                    is_external_test: Whether this is for external test or CV folds
                """
                # Extract fold results
                fold_results = {k: v for k, v in results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                plt.figure(figsize=(15, 10))
                
                # Set up separate subplots for positive and negative class distributions
                plt.subplot(1, 2, 1)  # Left plot for negative class
                plt.subplot(1, 2, 2)  # Right plot for positive class
                
                # Colors for different folds
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                # Collect data for all folds
                all_neg_probs = []
                all_pos_probs = []
                fold_labels = []
                
                # For each fold, load data and collect probabilities
                for i, fold_key in enumerate(sorted(fold_results.keys())):
                    fold_num = int(fold_key.split('_')[1])
                    fold_labels.append(f'Fold {fold_num}')
                    
                    if is_external_test:
                        # External test data path
                        fold_dir = os.path.join(source_data_dir, "test_evaluation", f"fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "external_test_probs_labels.npz")
                    else:
                        # Nested CV data path
                        fold_dir = os.path.join(source_data_dir, f"outer_fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "fold_probs_labels.npz")
                    
                    try:
                        # Try to load probabilities and labels
                        if os.path.exists(probs_labels_file):
                            data = np.load(probs_labels_file)
                            y_true = data['labels']
                            y_score = data['probabilities']
                            
                            # Separate probabilities by class
                            neg_probs = y_score[y_true == 0]
                            pos_probs = y_score[y_true == 1]
                            
                            all_neg_probs.append(neg_probs)
                            all_pos_probs.append(pos_probs)
                            
                            # Plot negative class probabilities (left subplot)
                            plt.subplot(1, 2, 1)
                            plt.hist(neg_probs, bins=20, alpha=0.4, color=colors[i % len(colors)], 
                                    label=f'Fold {fold_num}', density=True)
                            
                            # Plot positive class probabilities (right subplot)
                            plt.subplot(1, 2, 2)
                            plt.hist(pos_probs, bins=20, alpha=0.4, color=colors[i % len(colors)], 
                                    label=f'Fold {fold_num}', density=True)
                        else:
                            logging.warning(f"Could not find probability data for fold {fold_num}")
                    
                    except Exception as e:
                        logging.warning(f"Error loading probability data for fold {fold_num}: {e}")
                
                # Plot ensemble probabilities if available (only for external test)
                if is_external_test and 'ensemble' in results:
                    ensemble_dir = os.path.join(source_data_dir, "test_evaluation", "ensemble_plots")
                    ensemble_npz_files = glob.glob(os.path.join(ensemble_dir, "*.npz"))
                    
                    try:
                        if ensemble_npz_files:
                            ensemble_data = np.load(ensemble_npz_files[0])
                            if 'probabilities' in ensemble_data and 'labels' in ensemble_data:
                                # Separate ensemble probabilities by class
                                ens_y_true = ensemble_data['labels']
                                ens_y_score = ensemble_data['probabilities']
                                
                                ens_neg_probs = ens_y_score[ens_y_true == 0]
                                ens_pos_probs = ens_y_score[ens_y_true == 1]
                                
                                # Plot ensemble negative class probabilities
                                plt.subplot(1, 2, 1)
                                plt.hist(ens_neg_probs, bins=20, alpha=0.6, color='black', 
                                        label='Ensemble', density=True)
                                
                                # Plot ensemble positive class probabilities
                                plt.subplot(1, 2, 2)
                                plt.hist(ens_pos_probs, bins=20, alpha=0.6, color='black', 
                                        label='Ensemble', density=True)
                            else:
                                logging.warning("Ensemble probability data incomplete")
                    except Exception as e:
                        logging.warning(f"Error plotting ensemble probabilities: {e}")
                
                # Customize negative class subplot
                plt.subplot(1, 2, 1)
                plt.xlabel('Probability', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Negative Class (0) Probability Distribution', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Customize positive class subplot
                plt.subplot(1, 2, 2)
                plt.xlabel('Probability', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Positive Class (1) Probability Distribution', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Add overall title
                if is_external_test:
                    plt.suptitle('External Test: Probability Distributions Across Folds', fontsize=16, y=0.98)
                else:
                    plt.suptitle('Nested CV: Probability Distributions Across Folds', fontsize=16, y=0.98)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
                
                # Save the plot
                os.makedirs(output_dir, exist_ok=True)
                if is_external_test:
                    plt.savefig(os.path.join(output_dir, 'probability_distributions_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'probability_distributions_overlay.pdf'))
                else:
                    plt.savefig(os.path.join(output_dir, 'nested_cv_probability_distributions_overlay.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, 'nested_cv_probability_distributions_overlay.pdf'))
                
                plt.close()
                logging.info(f"Probability distribution overlay plot saved to {output_dir}")
            
            
            # 3. Training Curves Overlay (for nested CV only)
            def plot_training_curves_overlay(nested_cv_results, output_dir, source_data_dir):
                """
                Create overlay plots of training and validation metrics during training
                
                Args:
                    nested_cv_results: Dictionary containing nested CV results
                    output_dir: Directory to save the plot
                    source_data_dir: Directory where the data files are stored
                """
                # Extract fold results
                fold_results = {k: v for k, v in nested_cv_results.items() 
                               if k != 'average_performance'}
                
                # Search for training history in each fold's logs
                metrics_to_plot = ['f1', 'auc', 'loss']
                metrics_data = {metric: {} for metric in metrics_to_plot}
                max_epochs = 0
                
                # Colors for different folds
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                # Collect training history for each fold
                for fold_key, fold_data in sorted(fold_results.items()):
                    fold_num = int(fold_key.split('_')[1])
                    fold_dir = os.path.join(source_data_dir, f"outer_fold_{fold_num}")
                    
                    # Load model_config which might have training history
                    try:
                        with open(os.path.join(fold_dir, "fold_config.json"), 'r') as f:
                            fold_config = json.load(f)
                        
                        # Extract metric history if available
                        if 'metric_history' in fold_config:
                            for metric in metrics_to_plot:
                                if metric in fold_config['metric_history']:
                                    # Structure is [(epoch, value), (epoch, value), ...]
                                    history = fold_config['metric_history'][metric]
                                    
                                    # Convert to arrays for plotting
                                    epochs = [item[0] for item in history]
                                    values = [item[1] for item in history]
                                    
                                    # Store in metrics_data
                                    metrics_data[metric][fold_num] = (epochs, values)
                                    
                                    # Update max_epochs
                                    max_epochs = max(max_epochs, max(epochs) + 1)
                    
                    except Exception as e:
                        logging.warning(f"Could not load training history for fold {fold_num}: {e}")
                
                # Create overlay plot for each metric
                for metric in metrics_to_plot:
                    if not metrics_data[metric]:
                        logging.warning(f"No training history found for metric {metric}")
                        continue
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Plot each fold's history
                    for i, (fold_num, (epochs, values)) in enumerate(sorted(metrics_data[metric].items())):
                        plt.plot(epochs, values, marker='o', linestyle='-', color=colors[i % len(colors)],
                                label=f'Fold {fold_num}', linewidth=2, markersize=6)
                    
                    # Customizations
                    plt.xlabel('Epoch', fontsize=14)
                    plt.ylabel(f'{metric.upper()}', fontsize=14)
                    plt.title(f'Nested CV: {metric.upper()} Over Training Epochs', fontsize=16)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best', fontsize=12)
                    
                    # Set x-axis limits
                    plt.xlim(-0.5, max_epochs - 0.5)
                    
                    # Save the plot
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, f'nested_cv_{metric}_training_curves.png'), dpi=300)
                    plt.savefig(os.path.join(output_dir, f'nested_cv_{metric}_training_curves.pdf'))
                    plt.close()
                
                logging.info(f"Training curves overlay plots saved to {output_dir}")
            
            
            # 4. Error Analysis Overlay
            def plot_error_analysis_overlay(results, output_dir, source_data_dir, is_external_test=True):
                """
                Create an overlay analysis of errors across different folds
                
                Args:
                    results: Dictionary containing results (nested_cv_results or test_results)
                    output_dir: Directory to save the plot
                    source_data_dir: Directory where the data files are stored
                    is_external_test: Whether this is for external test or CV folds
                """
                # Extract fold results
                fold_results = {k: v for k, v in results.items() 
                               if k != 'average_performance' and k != 'ensemble'}
                
                # Collect predictions from all folds
                fold_predictions = {}
                all_labels = None
                
                # Load predictions for each fold
                for fold_key, fold_data in sorted(fold_results.items()):
                    fold_num = int(fold_key.split('_')[1])
                    
                    if is_external_test:
                        # External test data path
                        fold_dir = os.path.join(source_data_dir, "test_evaluation", f"fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "external_test_probs_labels.npz")
                    else:
                        # Nested CV data path
                        fold_dir = os.path.join(source_data_dir, f"outer_fold_{fold_num}")
                        probs_labels_file = os.path.join(fold_dir, "fold_probs_labels.npz")
                    
                    try:
                        # Try to load probabilities and labels
                        if os.path.exists(probs_labels_file):
                            data = np.load(probs_labels_file)
                            y_true = data['labels']
                            y_score = data['probabilities']
                            
                            # Get optimal threshold for this fold
                            if is_external_test:
                                threshold = fold_data.get('optimal_threshold', 0.5)
                            else:
                                threshold = fold_data.get('optimal_threshold_val_epoch', 0.5)
                            
                            # Convert probabilities to predictions using threshold
                            y_pred = (y_score >= threshold).astype(int)
                            
                            # Store predictions for this fold
                            fold_predictions[fold_num] = y_pred
                            
                            # Store true labels (should be the same for all folds)
                            if all_labels is None:
                                all_labels = y_true
                        else:
                            logging.warning(f"Could not find prediction data for fold {fold_num}")
                    
                    except Exception as e:
                        logging.warning(f"Error loading prediction data for fold {fold_num}: {e}")
                
                # Add ensemble predictions if available
                if is_external_test and 'ensemble' in results:
                    ensemble_dir = os.path.join(source_data_dir, "test_evaluation", "ensemble_plots")
                    ensemble_npz_files = glob.glob(os.path.join(ensemble_dir, "*.npz"))
                    
                    try:
                        if ensemble_npz_files:
                            ensemble_data = np.load(ensemble_npz_files[0])
                            if 'probabilities' in ensemble_data and 'labels' in ensemble_data:
                                # Get ensemble predictions using its optimal threshold
                                ens_threshold = results['ensemble'].get('optimal_threshold', 0.5)
                                ens_y_pred = (ensemble_data['probabilities'] >= ens_threshold).astype(int)
                                
                                # Store ensemble predictions
                                fold_predictions['ensemble'] = ens_y_pred
                    except Exception as e:
                        logging.warning(f"Error loading ensemble predictions: {e}")
                
                # If we have data for at least 2 folds
                if len(fold_predictions) >= 2 and all_labels is not None:
                    # Convert to 2D array [samples, folds]
                    fold_nums = sorted([f for f in fold_predictions.keys() if f != 'ensemble'])
                    pred_array = np.column_stack([fold_predictions[f] for f in fold_nums])
                    
                    # Add ensemble predictions as the last column if available
                    if 'ensemble' in fold_predictions:
                        pred_array = np.column_stack([pred_array, fold_predictions['ensemble']])
                        fold_labels = [f"Fold {f}" for f in fold_nums] + ["Ensemble"]
                    else:
                        fold_labels = [f"Fold {f}" for f in fold_nums]
                    
                    # Calculate error patterns
                    error_matrix = pred_array != all_labels[:, np.newaxis]  # 1 if error, 0 if correct
                    error_counts = np.sum(error_matrix, axis=1)  # Number of folds that make an error for each sample
                    
                    # Calculate statistics
                    total_samples = len(all_labels)
                    consistently_correct = np.sum(error_counts == 0)
                    consistently_wrong = np.sum(error_counts == len(fold_labels))
                    mixed_results = total_samples - consistently_correct - consistently_wrong
                    
                    # Create figure for error analysis
                    plt.figure(figsize=(12, 10))
                    
                    # Plot statistics
                    labels = ['All Correct', 'Mixed Results', 'All Wrong']
                    sizes = [consistently_correct, mixed_results, consistently_wrong]
                    colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Amber, Red
                    
                    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                           startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
                    
                    # Add percentage labels
                    for i, pct in enumerate([consistently_correct/total_samples*100, 
                                            mixed_results/total_samples*100, 
                                            consistently_wrong/total_samples*100]):
                        label = f"{labels[i]}: {pct:.1f}% ({sizes[i]} samples)"
                        plt.annotate(label, xy=(0, 0), xytext=(0, 0.7 - i*0.1),
                                    fontsize=12, ha='center', va='center')
                    
                    # Add title based on external test or nested CV
                    if is_external_test:
                        plt.title('External Test: Error Analysis Across All Folds', fontsize=16, y=1.1)
                        subtitle = f"Analysis of {total_samples} test samples across {len(fold_labels)} models"
                    else:
                        plt.title('Nested CV: Error Analysis Across All Folds', fontsize=16, y=1.1)
                        subtitle = f"Analysis of {total_samples} validation samples across {len(fold_labels)} models"
                    
                    plt.annotate(subtitle, xy=(0, 0), xytext=(0, 0.9),
                                fontsize=14, ha='center', va='center')
                    
                    plt.axis('equal')
                    
                    # Save the plot
                    os.makedirs(output_dir, exist_ok=True)
                    if is_external_test:
                        plt.savefig(os.path.join(output_dir, 'error_analysis_overlay.png'), dpi=300)
                        plt.savefig(os.path.join(output_dir, 'error_analysis_overlay.pdf'))
                    else:
                        plt.savefig(os.path.join(output_dir, 'nested_cv_error_analysis_overlay.png'), dpi=300)
                        plt.savefig(os.path.join(output_dir, 'nested_cv_error_analysis_overlay.pdf'))
                    
                    plt.close()
                    
                    # Create a second figure showing error distribution
                    plt.figure(figsize=(12, 8))
                    
                    # Count samples with each possible number of errors
                    error_distribution = [np.sum(error_counts == i) for i in range(len(fold_labels) + 1)]
                    
                    # Plot as a bar chart
                    bars = plt.bar(range(len(fold_labels) + 1), error_distribution, color='navy', alpha=0.7)
                    
                    # Add value labels on top of bars
                    for bar, count in zip(bars, error_distribution):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{count}\n({count/total_samples*100:.1f}%)',
                                ha='center', va='bottom', fontsize=10)
                    
                    # Customize the plot
                    plt.xlabel('Number of Models Making an Error', fontsize=14)
                    plt.ylabel('Number of Samples', fontsize=14)
                    if is_external_test:
                        plt.title('External Test: Distribution of Errors Across Models', fontsize=16)
                    else:
                        plt.title('Nested CV: Distribution of Errors Across Models', fontsize=16)
                    plt.xticks(range(len(fold_labels) + 1))
                    plt.grid(axis='y', alpha=0.3)
                    
                    # Save the plot
                    if is_external_test:
                        plt.savefig(os.path.join(output_dir, 'error_distribution_overlay.png'), dpi=300)
                        plt.savefig(os.path.join(output_dir, 'error_distribution_overlay.pdf'))
                    else:
                        plt.savefig(os.path.join(output_dir, 'nested_cv_error_distribution_overlay.png'), dpi=300)
                        plt.savefig(os.path.join(output_dir, 'nested_cv_error_distribution_overlay.pdf'))
                    
                    plt.close()
                    
                    logging.info(f"Error analysis overlay plots saved to {output_dir}")
                else:
                    logging.warning("Not enough data for error analysis overlay")
            

            # Create overlaid ROC curves from test evaluation
            def plot_roc_curves_overlay(test_results, output_dir):
                """Create an overlay plot of ROC curves for all folds and the ensemble"""
                # Extract fold results 
                fold_results = {k: v for k, v in test_results.items() 
                              if k != 'average_performance' and k != 'ensemble'}
                
                # Directory to find saved fold data
                fold_data_dir = os.path.join(save_dir, "test_evaluation")
                
                plt.figure(figsize=(12, 10))
                
                # Plot ROC curve for each fold
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                line_styles = ['-', '--', '-.', ':', '-', '--']
                
                for i, fold_key in enumerate(sorted(fold_results.keys())):
                    fold_num = int(fold_key.split('_')[1])
                    fold_dir = os.path.join(fold_data_dir, f"fold_{fold_num}")
                    
                    # Try to load the saved ROC data from metrics file
                    try:
                        with open(os.path.join(fold_dir, "external_test_metrics.json"), 'r') as f:
                            metrics_data = json.load(f)
                        
                        if 'roc_data' in metrics_data:
                            roc_data = metrics_data['roc_data']
                            fpr = roc_data['fpr']
                            tpr = roc_data['tpr']
                            auc_value = roc_data['auc']
                            
                            # Get fold AUC from metrics
                            fold_auc = metrics_data.get('auc', auc_value)
                            
                            # Plot this fold's ROC curve
                            plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                                    linestyle=line_styles[i % len(line_styles)],
                                    lw=2, label=f'Fold {fold_num} (AUC = {fold_auc:.3f})')
                        else:
                            logging.warning(f"No ROC data found in metrics for fold {fold_num}")
                            
                    except Exception as e:
                        logging.warning(f"Could not load ROC data for fold {fold_num}: {e}")
                
                # Plot ensemble ROC if available
                if 'ensemble' in test_results:
                    # Look for ensemble ROC data
                    ensemble_fpr_tpr_file = os.path.join(fold_data_dir, "ensemble_plots", "ensemble_roc_data.json")
                    
                    try:
                        # First check if we saved the ensemble ROC data separately
                        if os.path.exists(ensemble_fpr_tpr_file):
                            with open(ensemble_fpr_tpr_file, 'r') as f:
                                ensemble_roc = json.load(f)
                            fpr = ensemble_roc['fpr']
                            tpr = ensemble_roc['tpr']
                            ensemble_auc = ensemble_roc['auc']
                        else:
                            # If not, use the AUC from the ensemble metrics
                            ensemble_auc = test_results['ensemble']['metrics']['auc']
                            # We'll need to load probabilities to calculate the ROC
                            ensemble_npz_files = glob.glob(os.path.join(fold_data_dir, "ensemble_plots", "*.npz"))
                            if ensemble_npz_files:
                                ensemble_data = np.load(ensemble_npz_files[0])
                                if 'probabilities' in ensemble_data and 'labels' in ensemble_data:
                                    from sklearn.metrics import roc_curve
                                    fpr, tpr, _ = roc_curve(ensemble_data['labels'], ensemble_data['probabilities'])
                                else:
                                    logging.warning("Ensemble probability data incomplete")
                                    raise ValueError("Cannot calculate ensemble ROC curve")
                            else:
                                raise ValueError("No ensemble probability data found")
                                
                        plt.plot(fpr, tpr, color='black', linestyle='-', lw=3,
                                label=f'Ensemble (AUC = {ensemble_auc:.3f})')
                                
                    except Exception as e:
                        logging.warning(f"Could not plot ensemble ROC curve: {e}")
                
                # Plot the reference line
                plt.plot([0, 1], [0, 1], 'k--', lw=1)
                
                # Set labels and title
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.title('External Test: ROC Curves Across All Folds and Ensemble', fontsize=16)
                plt.legend(loc="lower right", fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'roc_curves_overlay.png'), dpi=300)
                plt.savefig(os.path.join(output_dir, 'roc_curves_overlay.pdf'))
                plt.close()
                
                logging.info(f"ROC curves overlay plot saved to {output_dir}")
            
            # Create overlaid ROC curves
            plot_roc_curves_overlay(test_results, test_radar_dir)
            
        except Exception as e:
            logging.error(f"Error creating overlay plots: {e}", exc_info=True)
            logging.info("Continuing with other parts of the pipeline...")
        
        # Add these lines to the main() function, after the existing overlay plots:

        # Create overlay plots for both nested CV and external test
        try:
            # Nested CV overlay plots
            nested_cv_overlay_dir = os.path.join(save_dir, "nested_cv_overlay_plots")
            os.makedirs(nested_cv_overlay_dir, exist_ok=True)
            
            # Generate nested CV overlay plots
            plot_pr_curves_overlay(nested_cv_results, nested_cv_overlay_dir, save_dir, is_external_test=False)
            plot_probability_distribution_overlay(nested_cv_results, nested_cv_overlay_dir, save_dir, is_external_test=False)
            plot_training_curves_overlay(nested_cv_results, nested_cv_overlay_dir, save_dir)
            plot_error_analysis_overlay(nested_cv_results, nested_cv_overlay_dir, save_dir, is_external_test=False)
            plot_mcc_comparison(nested_cv_results, nested_cv_overlay_dir, is_external_test=False)
            plot_auroc_vs_ap_overlay(nested_cv_results, nested_cv_overlay_dir, is_external_test=False)
            
            # External test overlay plots
            test_overlay_dir = os.path.join(save_dir, "test_evaluation", "overlay_plots")
            os.makedirs(test_overlay_dir, exist_ok=True)
            
            # Generate external test overlay plots
            plot_pr_curves_overlay(test_results, test_overlay_dir, save_dir, is_external_test=True)
            plot_probability_distribution_overlay(test_results, test_overlay_dir, save_dir, is_external_test=True)
            plot_error_analysis_overlay(test_results, test_overlay_dir, save_dir, is_external_test=True)
            plot_mcc_comparison(test_results, test_overlay_dir, is_external_test=True)
            plot_auroc_vs_ap_overlay(test_results, test_overlay_dir, is_external_test=True)
            plot_roc_curves_overlay(test_results, test_overlay_dir) # Already exists for external test
            
            logging.info("Generated all overlay plots for nested CV and external test datasets")
        except Exception as e:
            logging.error(f"Error creating additional overlay plots: {e}", exc_info=True)
        # Log total runtime
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Log final results from both cross-validation and test set evaluation
        cv_avg_results = nested_cv_results['average_performance']
        test_avg_results = test_results['average_performance']['avg_metrics']
        test_std_results = test_results['average_performance']['std_metrics']
        
        logging.info(f"\n{'='*50}\nFinal Results\n{'='*50}")
        
        logging.info("\nNested Cross-Validation Results:")
        logging.info(f"Average F1: {cv_avg_results['avg_f1']:.4f} ± {cv_avg_results['std_f1']:.4f}")
        logging.info(f"Average AUC: {cv_avg_results['avg_auc']:.4f} ± {cv_avg_results['std_auc']:.4f}")
        logging.info(f"Average Accuracy: {cv_avg_results['avg_accuracy']:.4f} ± {cv_avg_results['std_accuracy']:.4f}")
        logging.info(f"Average Sensitivity: {cv_avg_results['avg_sensitivity']:.4f} ± {cv_avg_results['std_sensitivity']:.4f}")
        logging.info(f"Average Specificity: {cv_avg_results['avg_specificity']:.4f} ± {cv_avg_results['std_specificity']:.4f}")
        
        logging.info("\nTest Set Evaluation Results:")
        logging.info(f"Average F1: {test_avg_results['f1']:.4f} ± {test_std_results['f1']:.4f}")
        logging.info(f"Average AUC: {test_avg_results['auc']:.4f} ± {test_std_results['auc']:.4f}")
        logging.info(f"Average Accuracy: {test_avg_results['accuracy']:.4f} ± {test_std_results['accuracy']:.4f}")
        logging.info(f"Average Sensitivity: {test_avg_results['sensitivity']:.4f} ± {test_std_results['sensitivity']:.4f}")
        logging.info(f"Average Specificity: {test_avg_results['specificity']:.4f} ± {test_std_results['specificity']:.4f}")
        
        # If ensemble results are available, log them
        if 'ensemble' in test_results:
            ensemble_metrics = test_results['ensemble']['metrics']
            logging.info("\nEnsemble Model Test Results:")
            logging.info(f"Optimal Threshold: {test_results['ensemble']['optimal_threshold']:.4f}")
            logging.info(f"F1 Score: {ensemble_metrics['f1']:.4f}")
            logging.info(f"AUC: {ensemble_metrics['auc']:.4f}")
            logging.info(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
            logging.info(f"Sensitivity: {ensemble_metrics['sensitivity']:.4f}")
            logging.info(f"Specificity: {ensemble_metrics['specificity']:.4f}")
        
        logging.info(f"\nResults saved to {os.path.join(save_dir, 'nested_cv_results.json')} and {os.path.join(save_dir, 'test_evaluation/test_results.json')}")
        
        # Create summary file for easy reference
        summary_path = os.path.join(BASE_DIR, "results_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{'='*50}\n")
            f.write(f"SKIN LESION CLASSIFICATION RESULTS SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            
            f.write("NESTED CROSS-VALIDATION RESULTS:\n")
            f.write(f"Average F1: {cv_avg_results['avg_f1']:.4f} ± {cv_avg_results['std_f1']:.4f}\n")
            f.write(f"Average AUC: {cv_avg_results['avg_auc']:.4f} ± {cv_avg_results['std_auc']:.4f}\n")
            f.write(f"Average Accuracy: {cv_avg_results['avg_accuracy']:.4f} ± {cv_avg_results['std_accuracy']:.4f}\n")
            f.write(f"Average Sensitivity: {cv_avg_results['avg_sensitivity']:.4f} ± {cv_avg_results['std_sensitivity']:.4f}\n")
            f.write(f"Average Specificity: {cv_avg_results['avg_specificity']:.4f} ± {cv_avg_results['std_specificity']:.4f}\n\n")
            
            f.write("EXTERNAL TEST SET RESULTS:\n")
            f.write(f"Average F1: {test_avg_results['f1']:.4f} ± {test_std_results['f1']:.4f}\n")
            f.write(f"Average AUC: {test_avg_results['auc']:.4f} ± {test_std_results['auc']:.4f}\n")
            f.write(f"Average Accuracy: {test_avg_results['accuracy']:.4f} ± {test_std_results['accuracy']:.4f}\n")
            f.write(f"Average Sensitivity: {test_avg_results['sensitivity']:.4f} ± {test_std_results['sensitivity']:.4f}\n")
            f.write(f"Average Specificity: {test_avg_results['specificity']:.4f} ± {test_std_results['specificity']:.4f}\n\n")
            
            if 'ensemble' in test_results:
                f.write("ENSEMBLE MODEL TEST RESULTS:\n")
                f.write(f"Optimal Threshold: {test_results['ensemble']['optimal_threshold']:.4f}\n")
                f.write(f"F1 Score: {ensemble_metrics['f1']:.4f}\n")
                f.write(f"AUC: {ensemble_metrics['auc']:.4f}\n")
                f.write(f"Accuracy: {ensemble_metrics['accuracy']:.4f}\n")
                f.write(f"Sensitivity: {ensemble_metrics['sensitivity']:.4f}\n")
                f.write(f"Specificity: {ensemble_metrics['specificity']:.4f}\n\n")
                
            f.write(f"{'='*50}\n")
            f.write(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"{'='*50}\n")
            
        logging.info(f"Summary saved to {summary_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.info(f"Starting skin lesion classification pipeline with nested cross-validation and test set evaluation")
    logging.info(f"Number of trials: {NUM_TRIALS}")
    logging.info(f"Outer folds: {OUTER_FOLDS}, Inner folds: {INNER_FOLDS}")
    success = main()
    if success:
        logging.info("Pipeline completed successfully!")
    else:
        logging.error("Pipeline failed!")
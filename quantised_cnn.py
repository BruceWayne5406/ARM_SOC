import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
import torch.nn.functional as F
from tqdm import tqdm
import os
import urllib.request
import zipfile
from pathlib import Path

# Import quantization modules
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

# Define the 3-layer CNN with quantization support
class ThreeLayerCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ThreeLayerCNN, self).__init__()

        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.pool1(F.relu(self.conv1(x)))

        # Layer 2
        x = self.pool2(F.relu(self.conv2(x)))

        # Layer 3
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Quantization-ready version of the CNN
class ThreeLayerCNN_Quantizable(nn.Module):
    def __init__(self, num_classes=80):
        super(ThreeLayerCNN_Quantizable, self).__init__()

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Dequantize output
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv-ReLU layers for better quantization"""
        torch.quantization.fuse_modules(self, [['conv1', 'relu1']], inplace=True)
        torch.quantization.fuse_modules(self, [['conv2', 'relu2']], inplace=True)
        torch.quantization.fuse_modules(self, [['conv3', 'relu3']], inplace=True)
        torch.quantization.fuse_modules(self, [['fc1', 'relu4']], inplace=True)


# Download progress callback
def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100.0 / total_size, 100)
    print(f'\rDownloading... {percent:.1f}%', end='')


# Function to download and extract COCO dataset
def download_coco_dataset(data_dir='./coco_data'):
    """
    Downloads COCO 2017 dataset including images and annotations.
    This will download ~20GB of data.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # COCO dataset URLs
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    files = {
        'train_images': data_dir / 'train2017.zip',
        'val_images': data_dir / 'val2017.zip',
        'annotations': data_dir / 'annotations_trainval2017.zip'
    }

    # Download files
    for key, url in urls.items():
        file_path = files[key]

        if file_path.exists():
            print(f'{key} already downloaded.')
            continue

        print(f'\nDownloading {key} from {url}')
        print(f'This may take a while...')

        try:
            urllib.request.urlretrieve(url, file_path, download_progress)
            print(f'\n{key} downloaded successfully!')
        except Exception as e:
            print(f'\nError downloading {key}: {e}')
            return False

    # Extract files
    print('\nExtracting files...')
    for key, file_path in files.items():
        extract_dir = data_dir

        # Check if already extracted
        if key == 'train_images' and (data_dir / 'train2017').exists():
            print(f'{key} already extracted.')
            continue
        if key == 'val_images' and (data_dir / 'val2017').exists():
            print(f'{key} already extracted.')
            continue
        if key == 'annotations' and (data_dir / 'annotations').exists():
            print(f'{key} already extracted.')
            continue

        print(f'Extracting {key}...')
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f'{key} extracted successfully!')
        except Exception as e:
            print(f'Error extracting {key}: {e}')
            return False

    print('\n✓ COCO dataset ready!')
    return True


# Custom collate function for COCO dataset
def coco_collate_fn(batch):
    images = []
    labels = []

    for img, target in batch:
        images.append(img)
        # Extract category ids from annotations
        if len(target) > 0:
            cats = [ann['category_id'] for ann in target]
            # Create multi-label target (80 classes in COCO)
            label = torch.zeros(80)
            for cat in cats:
                if cat > 0 and cat <= 80:
                    label[cat - 1] = 1
        else:
            label = torch.zeros(80)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


# Quantization-Aware Training function
def train_qat(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """Train model with Quantization-Aware Training"""
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'QAT Epoch {epoch+1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f'QAT Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


# Evaluation function
def evaluate_model(model, val_loader, device, quantized=False):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Evaluating')
        for images, labels in progress_bar:
            if not quantized:
                images, labels = images.to(device), labels.to(device)
            else:
                # For quantized models, keep on CPU
                images, labels = images, labels

            outputs = model(images)

            # For multi-label classification
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


# Post-Training Static Quantization
def quantize_model_static(model, val_loader):
    """
    Apply post-training static quantization (PTQ)
    Best for FPGA deployment - uses calibration data
    """
    print("\n" + "="*60)
    print("POST-TRAINING STATIC QUANTIZATION (PTQ)")
    print("="*60)

    # Create quantizable model and load weights
    model_fp32 = ThreeLayerCNN_Quantizable(num_classes=80)

    # Copy weights from original model
    if isinstance(model, ThreeLayerCNN):
        model_fp32.conv1.weight.data = model.conv1.weight.data.clone()
        model_fp32.conv1.bias.data = model.conv1.bias.data.clone()
        model_fp32.conv2.weight.data = model.conv2.weight.data.clone()
        model_fp32.conv2.bias.data = model.conv2.bias.data.clone()
        model_fp32.conv3.weight.data = model.conv3.weight.data.clone()
        model_fp32.conv3.bias.data = model.conv3.bias.data.clone()
        model_fp32.fc1.weight.data = model.fc1.weight.data.clone()
        model_fp32.fc1.bias.data = model.fc1.bias.data.clone()
        model_fp32.fc2.weight.data = model.fc2.weight.data.clone()
        model_fp32.fc2.bias.data = model.fc2.bias.data.clone()

    model_fp32.eval()

    # Fuse layers
    model_fp32.fuse_model()

    # Specify quantization configuration
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model_fp32)

    # Calibrate with representative data
    print("Calibrating with validation data...")
    model_prepared.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(val_loader, desc="Calibration")):
            if i >= 100:  # Use 100 batches for calibration
                break
            model_prepared(images)

    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)

    print("✓ Static quantization complete!")
    return model_quantized


# Quantization-Aware Training (QAT)
def quantize_model_qat(model, train_loader, val_loader, device, num_epochs=5):
    """
    Apply Quantization-Aware Training
    Better accuracy but requires retraining
    """
    print("\n" + "="*60)
    print("QUANTIZATION-AWARE TRAINING (QAT)")
    print("="*60)

    # Create quantizable model and load weights
    model_fp32 = ThreeLayerCNN_Quantizable(num_classes=80)

    # Copy weights from original model
    if isinstance(model, ThreeLayerCNN):
        model_fp32.conv1.weight.data = model.conv1.weight.data.clone()
        model_fp32.conv1.bias.data = model.conv1.bias.data.clone()
        model_fp32.conv2.weight.data = model.conv2.weight.data.clone()
        model_fp32.conv2.bias.data = model.conv2.bias.data.clone()
        model_fp32.conv3.weight.data = model.conv3.weight.data.clone()
        model_fp32.conv3.bias.data = model.conv3.bias.data.clone()
        model_fp32.fc1.weight.data = model.fc1.weight.data.clone()
        model_fp32.fc1.bias.data = model.fc1.bias.data.clone()
        model_fp32.fc2.weight.data = model.fc2.weight.data.clone()
        model_fp32.fc2.bias.data = model.fc2.bias.data.clone()

    model_fp32 = model_fp32.to(device)

    # Fuse layers
    model_fp32.fuse_model()

    # Specify quantization configuration
    model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Prepare for QAT
    model_prepared = torch.quantization.prepare_qat(model_fp32)

    # Train with QAT
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_prepared.parameters(), lr=0.0001)

    train_qat(model_prepared, train_loader, criterion, optimizer, device, num_epochs)

    # Convert to quantized model
    model_prepared.eval()
    model_prepared.cpu()
    model_quantized = torch.quantization.convert(model_prepared)

    print("✓ QAT complete!")
    return model_quantized


# Export quantized model for FPGA
def export_quantized_model(model, filepath='quantized_model_int8.pth'):
    """
    Export quantized model with INT8 weights
    """
    # Save the quantized model
    torch.save(model.state_dict(), filepath)
    print(f"\n✓ Quantized model saved to: {filepath}")

    # Save model size comparison
    print("\nModel Size Comparison:")

    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"Quantized model size: {file_size / (1024*1024):.2f} MB")

    return filepath


# Analyze quantization statistics
def analyze_quantization(model_fp32, model_quantized, val_loader):
    """
    Analyze the effects of quantization
    """
    print("\n" + "="*60)
    print("QUANTIZATION ANALYSIS")
    print("="*60)

    # Evaluate both models
    print("\nEvaluating FP32 model...")
    acc_fp32 = evaluate_model(model_fp32, val_loader, 'cpu', quantized=False)

    print("\nEvaluating INT8 quantized model...")
    acc_int8 = evaluate_model(model_quantized, val_loader, 'cpu', quantized=True)

    print("\n" + "-"*60)
    print(f"FP32 Accuracy:  {acc_fp32:.2f}%")
    print(f"INT8 Accuracy:  {acc_int8:.2f}%")
    print(f"Accuracy Drop:  {acc_fp32 - acc_int8:.2f}%")
    print("-"*60)

    # Model size comparison
    print("\nModel Size Analysis:")
    param_count = sum(p.numel() for p in model_fp32.parameters())
    fp32_size_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
    int8_size_mb = param_count * 1 / (1024 * 1024)  # 1 byte per int8

    print(f"FP32 Model Size (estimated): {fp32_size_mb:.2f} MB")
    print(f"INT8 Model Size (estimated): {int8_size_mb:.2f} MB")
    print(f"Compression Ratio: {fp32_size_mb/int8_size_mb:.2f}x")


# Main execution
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data directory
    data_dir = './coco_data'

    # Check if we should download the dataset
    print("Checking for COCO dataset...")

    # Check if dataset exists
    dataset_exists = (
        Path(f'{data_dir}/train2017').exists() and
        Path(f'{data_dir}/val2017').exists() and
        Path(f'{data_dir}/annotations').exists()
    )

    if not dataset_exists:
        print("NOTE: This will download ~20GB of data. Make sure you have enough disk space!")
        user_input = input("\nDo you want to download the COCO dataset? (yes/no): ").lower()

        if user_input in ['yes', 'y']:
            if not download_coco_dataset(data_dir):
                print("Failed to download dataset. Exiting.")
                exit(1)
        else:
            print("Dataset not found and download declined. Exiting.")
            exit(1)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Dataset paths
    train_img_dir = f'{data_dir}/train2017'
    train_ann_file = f'{data_dir}/annotations/instances_train2017.json'
    val_img_dir = f'{data_dir}/val2017'
    val_ann_file = f'{data_dir}/annotations/instances_val2017.json'

    # Load COCO dataset
    print("\nLoading COCO dataset...")
    try:
        train_dataset = CocoDetection(root=train_img_dir,
                                      annFile=train_ann_file,
                                      transform=transform)
        val_dataset = CocoDetection(root=val_img_dir,
                                    annFile=val_ann_file,
                                    transform=transform)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset was downloaded correctly.")
        exit(1)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32,
                             shuffle=True, collate_fn=coco_collate_fn,
                             num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32,
                           shuffle=False, collate_fn=coco_collate_fn,
                           num_workers=4)

    # Ask user which approach to use
    print("\n" + "="*60)
    print("QUANTIZATION OPTIONS")
    print("="*60)
    print("1. Train new FP32 model, then quantize (full workflow)")
    print("2. Load existing model and apply Post-Training Quantization (PTQ)")
    print("3. Load existing model and apply Quantization-Aware Training (QAT)")
    print("="*60)

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == '1':
        # Full workflow: Train then quantize
        print("\n" + "="*60)
        print("TRAINING FP32 MODEL")
        print("="*60)

        model = ThreeLayerCNN(num_classes=80).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nModel Architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train the model
        print("\nStarting training...")
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

        # Save FP32 model
        torch.save(model.state_dict(), 'coco_3layer_cnn_fp32.pth')
        print("\nFP32 Model saved as 'coco_3layer_cnn_fp32.pth'")

        # Evaluate FP32 model
        print("\nEvaluating FP32 model...")
        evaluate_model(model, val_loader, device)

        # Move model to CPU for quantization
        model.cpu()

        # Apply PTQ
        model_quantized = quantize_model_static(model, val_loader)

        # Analyze quantization
        analyze_quantization(model, model_quantized, val_loader)

        # Export quantized model
        export_quantized_model(model_quantized, 'coco_3layer_cnn_int8_ptq.pth')

    elif choice == '2':
        # Load existing model and apply PTQ
        model_path = input("\nEnter path to FP32 model (or press Enter for 'coco_3layer_cnn.pth'): ").strip()
        if not model_path:
            model_path = 'coco_3layer_cnn.pth'

        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            exit(1)

        print(f"\nLoading model from {model_path}...")
        model = ThreeLayerCNN(num_classes=80)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Apply PTQ
        model_quantized = quantize_model_static(model, val_loader)

        # Analyze quantization
        analyze_quantization(model, model_quantized, val_loader)

        # Export quantized model
        export_quantized_model(model_quantized, 'coco_3layer_cnn_int8_ptq.pth')

    elif choice == '3':
        # Load existing model and apply QAT
        model_path = input("\nEnter path to FP32 model (or press Enter for 'coco_3layer_cnn.pth'): ").strip()
        if not model_path:
            model_path = 'coco_3layer_cnn.pth'

        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            exit(1)

        print(f"\nLoading model from {model_path}...")
        model = ThreeLayerCNN(num_classes=80)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Apply QAT
        num_qat_epochs = int(input("\nEnter number of QAT epochs (recommended: 3-5): ") or "5")
        model_quantized = quantize_model_qat(model, train_loader, val_loader, device, num_qat_epochs)

        # Analyze quantization
        analyze_quantization(model, model_quantized, val_loader)

        # Export quantized model
        export_quantized_model(model_quantized, 'coco_3layer_cnn_int8_qat.pth')

    else:
        print("Invalid choice!")
        exit(1)

    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE!")
    print("="*60)
    
    print("1. The quantized model uses INT8 weights and activations")
    
   
    
    print("4. The model is optimized for:")
    print("   - 4x memory reduction")
    print("   - Faster inference on INT8-optimized hardware")
    print("   - Lower power consumption")
    print("="*60)

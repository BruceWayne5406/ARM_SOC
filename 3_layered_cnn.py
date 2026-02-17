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

# Define the 3-layer CNN
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
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

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

# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # For multi-label classification
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

# Main execution
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data directory
    data_dir = './coco_data'

    # Download COCO dataset
    print("Checking for COCO dataset...")
    print("NOTE: This will download ~20GB of data. Make sure you have enough disk space!")

    user_input = input("\nDo you want to download the COCO dataset? (yes/no): ").lower()

    if user_input in ['yes', 'y']:
        if not download_coco_dataset(data_dir):
            print("Failed to download dataset. Exiting.")
            exit(1)
    else:
        print("Skipping download. Make sure the dataset is already present in './coco_data/'")

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

    # Initialize model
    model = ThreeLayerCNN(num_classes=80).to(device)

    # Loss and optimizer (BCEWithLogitsLoss for multi-label classification)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'coco_3layer_cnn.pth')
    print("\nModel saved as 'coco_3layer_cnn.pth'")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# ---------------------- Define Dataset with Enhancement ----------------------
class DisasterDataset(Dataset):
    def __init__(self, root_dir, transform=None, enhance=True):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  
        self.image_pairs = []
        self.labels = []
        self.transform = transform
        self.enhance = enhance

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                event_folders = os.listdir(class_path)
                for event_folder in event_folders:
                    event_path = os.path.join(class_path, event_folder)
                    if os.path.isdir(event_path):
                        pre_img_path = os.path.join(event_path, 'pre_disaster.png')
                        post_img_path = os.path.join(event_path, 'post_disaster.png')
                        if os.path.exists(pre_img_path) and os.path.exists(post_img_path):
                            self.image_pairs.append((pre_img_path, post_img_path))
                            self.labels.append(idx)  # 0: destroyed, 1: major, 2: minor, 3: no

        # Convert labels to tensors
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_pairs)

    def _enhance_image(self, image):
        if not self.enhance:
            return image
        # Apply some common enhancement techniques - experiment with these!
        enhancer_brightness = ImageEnhance.Brightness(image)
        image = enhancer_brightness.enhance(1.1) # Slightly increase brightness

        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(1.1)   # Slightly increase contrast

        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(1.2)  # Slightly increase sharpness

        return image

    def __getitem__(self, idx):
        pre_img_path, post_img_path = self.image_pairs[idx]
        pre_image = Image.open(pre_img_path).convert('RGB')
        post_image = Image.open(post_img_path).convert('RGB')

        # Enhance the images
        pre_image = self._enhance_image(pre_image)
        post_image = self._enhance_image(post_image)

        label = self.labels[idx]

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)

        return pre_image, post_image, label

# ---------------------- Data Loading and Splitting ----------------------
root_dir = r"C:\Users\suhas\OneDrive\Desktop\Natural Disaster Managemant\building_image_pairs_8000_mp"  # Replace with your actual path
image_size = 224  # Standard size for EfficientNet

# Define transformations (applied AFTER enhancement)
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the full dataset with enhancement enabled for training
full_dataset = DisasterDataset(root_dir, transform=train_transform, enhance=True)

# Split into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)) # Added seed for reproducibility

# For the validation set, we use a separate dataset with enhancement disabled
val_dataset_no_enhance = DisasterDataset(root_dir, transform=val_transform, enhance=False)
_, val_dataset_no_enhance = train_test_split(val_dataset_no_enhance, test_size=val_size/len(full_dataset), random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset_no_enhance, batch_size=32, shuffle=False, num_workers=2)

# ---------------------- Define Model ----------------------
class SiameseEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(SiameseEfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b1(pretrained=True)
        # Remove the last classifier layer
        self.efficientnet.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 512), # Concatenate features from both images
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward_once(self, x):
        return self.efficientnet(x)

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        combined_features = torch.cat((feat1, feat2), dim=1)
        return self.fc(combined_features)

model = SiameseEfficientNet(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------- Loss and Optimizer ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose=True)

# ---------------------- Training Loop ----------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for pre_images, post_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training with Enhancement)"):
        pre_images = pre_images.to(device)
        post_images = post_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pre_images, post_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * pre_images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    # ---------------------- Validation Loop ----------------------
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for pre_images, post_images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation on Original Quality)"):
            pre_images = pre_images.to(device)
            post_images = post_images.to(device)
            labels = labels.to(device)

            outputs = model(pre_images, post_images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * pre_images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_val_loss = val_loss / len(val_dataset)
    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    scheduler.step(epoch_val_loss)

print("Finished Training")

# Save the model
torch.save(model.state_dict(), r'C:\Users\suhas\OneDrive\Desktop\Natural Disaster Managemant\models.pth')
print('Model saved to Specified Folder')
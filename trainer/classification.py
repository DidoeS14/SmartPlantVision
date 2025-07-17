import os
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch


class ClassificationTrainer:
    def __init__(self, path_to_dataset: str, epochs: int = 5):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.path = path_to_dataset
        self.epochs = epochs
        self.val_loader = None
        self.train_loader = None
        self.class_names = None
        self.model = None
        self.device = None
        self.criterion = None
        self.optimizer = None

    def run(self):
        self.logger.debug("Starting training pipeline...")
        self.load_dataset()
        self.prepare()
        self.train()

    def load_dataset(self):
        self.logger.debug(f"Loading dataset from: {self.path}")
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        # Load all data
        # full_dataset = datasets.ImageFolder("dataset/fruits_and_vegies", transform=transform)
        full_dataset = datasets.ImageFolder(self.path, transform=transform)

        # Class names
        self.class_names = full_dataset.classes
        self.logger.debug(f"Detected classes: {self.class_names}")

        # Split
        val_pct = 0.2
        val_size = int(len(full_dataset) * val_pct)
        train_size = len(full_dataset) - val_size
        self.logger.debug(f"Splitting dataset: {train_size} training, {val_size} validation")

        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        # DataLoaders
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        self.logger.debug("Data loaders created")

    def prepare(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Using device: {self.device}")

        # Load pretrained model and modify
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))  # adapt to your classes
        self.model = self.model.to(self.device)

        self.logger.debug("Model loaded and modified")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.logger.debug("Loss function and optimizer set")

    def train(self):
        self.logger.debug("Starting training loop...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    def save(self, save_path: str = 'output', name: str = 'model.pth'):
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, name)
        torch.save(self.model.state_dict(), full_path)
        self.logger.info(f"Model saved to {full_path}")

        # Test loading back
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.class_names))
        model.load_state_dict(torch.load(full_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        self.logger.debug("Model reloaded and set to eval mode")


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ct = ClassificationTrainer(path_to_dataset='../datasets/fruits_and_vegies')
    ct.run()
    ct.save()
import os
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch


class ClassificationTrainer:
    def __init__(self, path_to_dataset: str, epochs: int = 5):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.path = path_to_dataset
        self.epochs = epochs
        self.val_loader = None
        self.train_loader = None
        self.class_names = None
        self.model = None
        self.device = None
        self.criterion = None
        self.optimizer = None

    def run(self):
        self.logger.debug("Starting training pipeline...")
        self.load_dataset()
        self.prepare()
        self.train()

    def load_dataset(self):
        self.logger.debug(f"Loading dataset from: {self.path}")
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        # Load all data
        # full_dataset = datasets.ImageFolder("dataset/fruits_and_vegies", transform=transform)
        full_dataset = datasets.ImageFolder(self.path, transform=transform)

        # Class names
        self.class_names = full_dataset.classes
        self.logger.debug(f"Detected classes: {self.class_names}")

        # Split
        val_pct = 0.2
        val_size = int(len(full_dataset) * val_pct)
        train_size = len(full_dataset) - val_size
        self.logger.debug(f"Splitting dataset: {train_size} training, {val_size} validation")

        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        # DataLoaders
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        self.logger.debug("Data loaders created")

    def prepare(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Using device: {self.device}")

        # Load pretrained model and modify
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))  # adapt to your classes
        self.model = self.model.to(self.device)

        self.logger.debug("Model loaded and modified")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.logger.debug("Loss function and optimizer set")

    def train(self):
        self.logger.debug("Starting training loop...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    def save(self, save_path: str = 'output', name: str = 'model.pth'):
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, name)
        torch.save(self.model.state_dict(), full_path)
        self.logger.info(f"Model saved to {full_path}")

        # Test loading back
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.class_names))
        model.load_state_dict(torch.load(full_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        self.logger.debug("Model reloaded and set to eval mode")


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    ct = ClassificationTrainer(path_to_dataset='../datasets/fruits_and_vegies')
    ct.run()
    ct.save()

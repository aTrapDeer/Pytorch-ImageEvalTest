import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

def train():
    print("Starting training process...")

    # Define transforms
    print("Defining data transforms...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load training data
    print("Loading training data...")
    train_data_dir = "dataset/train"
    train_dataset = ImageFolder(root=train_data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Number of training samples: {len(train_dataset)}")

    # Load evaluation data
    print("Loading evaluation data...")
    eval_data_dir = "dataset/eval"
    eval_dataset = ImageFolder(root=eval_data_dir, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=32)
    print(f"Number of evaluation samples: {len(eval_dataset)}")

    # Define model and move to CPU/GPU
    print("Setting up the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = resnet18(pretrained=True)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    print("Starting training loop...")
    for epoch in range(10):  # Adjust number of epochs as needed
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate the model
        print(f"Evaluating model after epoch {epoch + 1}...")
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

    print("Training completed.")
    # Save the trained model
    print("Saving the trained model...")
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import CNN

# 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()         # 이전 기울기 초기화
        outputs = model(images)       # 순전파
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()               # 역전파
        optimizer.step()              # 가중치 업데이트

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100


def main():
    print(f"사용 디바이스: {DEVICE}")
    train_loader, test_loader = load_data()

    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | 정확도: {acc:.2f}%")

    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    print("모델 저장 완료: models/mnist_cnn.pth")


if __name__ == "__main__":
    main()

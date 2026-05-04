import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import CNN

# ── 하이퍼파라미터 ────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 20           # CIFAR-10은 MNIST보다 복잡해서 더 많은 에폭 필요
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 클래스 이름 (인덱스 0~9 순서)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_data():
    # 학습용 전처리: 데이터 증강(augmentation) 추가
    # CIFAR-10은 MNIST보다 복잡하므로 다양한 변형을 줘서 과적합 방지
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 50% 확률로 좌우 반전 (비행기가 뒤집혀도 비행기)
        transforms.RandomCrop(32, padding=4),  # 상하좌우 4픽셀 패딩 후 32×32로 랜덤 크롭
        transforms.ToTensor(),
        # CIFAR-10 전체 데이터셋의 RGB 채널별 평균/표준편차로 정규화
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 테스트용 전처리: 증강 없이 정규화만 (실제 예측 환경과 동일하게)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

    # 학습률 스케줄러: 10 에폭마다 학습률을 0.1배로 줄임
    # 초반엔 크게 학습하고, 후반엔 세밀하게 조정해 더 높은 정확도 도달
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        scheduler.step()
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss:.4f} | 정확도: {acc:.2f}%")

        # 가장 높은 정확도일 때만 저장 (마지막이 아닌 최고 성능 모델 보존)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/cifar10_cnn.pth")

    print(f"\n학습 완료 | 최고 정확도: {best_acc:.2f}%")
    print("모델 저장 완료: models/cifar10_cnn.pth")


if __name__ == "__main__":
    main()

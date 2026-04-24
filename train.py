import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import CNN

# ── 하이퍼파라미터 ────────────────────────────────────────────────────
# 모델 성능과 학습 방식을 결정하는 설정값들
BATCH_SIZE = 64       # 한 번에 학습할 이미지 수 (클수록 빠르지만 메모리 많이 사용)
EPOCHS = 5            # 전체 데이터를 몇 번 반복 학습할지
LEARNING_RATE = 0.001 # 가중치를 얼마나 크게 업데이트할지 (너무 크면 발산, 너무 작으면 느림)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 있으면 GPU, 없으면 CPU


def load_data():
    # 이미지 전처리 파이프라인 정의
    transform = transforms.Compose([
        transforms.ToTensor(),                        # PIL 이미지(0~255) → 텐서(0.0~1.0)로 변환
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST 데이터셋 전체의 평균/표준편차로 정규화
                                                      # → 값 범위를 균일하게 만들어 학습 안정화
    ])

    # MNIST 데이터셋 다운로드 및 로드
    # train=True: 학습용 60,000장 / train=False: 테스트용 10,000장
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # DataLoader: 데이터를 배치 단위로 묶어서 모델에 공급
    # shuffle=True: 매 에폭마다 순서를 섞어 특정 패턴에 과적합되는 것 방지
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train(model, loader, optimizer, criterion):
    model.train()  # 학습 모드: Dropout 등이 활성화됨
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()               # 이전 배치의 기울기 초기화 (누적되지 않도록)
        outputs = model(images)             # 순전파: 이미지 → 모델 → 예측값(10개 점수)
        loss = criterion(outputs, labels)   # 손실 계산: 예측값과 실제 정답의 차이
        loss.backward()                     # 역전파: 손실을 줄이기 위한 각 가중치의 기울기 계산
        optimizer.step()                    # 가중치 업데이트: 기울기 방향으로 가중치 조정

        total_loss += loss.item()

    return total_loss / len(loader)  # 에폭 평균 손실 반환


def evaluate(model, loader):
    model.eval()  # 평가 모드: Dropout 비활성화 (일관된 결과를 위해)
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 시에는 기울기 계산 불필요 → 메모리/속도 절약
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 10개 점수 중 가장 높은 인덱스 = 예측 숫자
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total * 100  # 정확도(%) 반환


def main():
    print(f"사용 디바이스: {DEVICE}")
    train_loader, test_loader = load_data()

    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Adam: 학습률을 가중치마다 자동으로 조절하는 옵티마이저 (SGD보다 대체로 빠르게 수렴)
    criterion = nn.CrossEntropyLoss()
    # CrossEntropyLoss: 다중 분류에 적합한 손실 함수
    # 모델의 출력(logit)을 확률로 변환(Softmax) 후 정답 클래스의 확률이 높을수록 손실이 낮아짐

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | 정확도: {acc:.2f}%")

    # 학습된 가중치만 저장 (모델 구조는 코드에 있으므로 가중치만 있으면 복원 가능)
    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    print("모델 저장 완료: models/mnist_cnn.pth")


if __name__ == "__main__":
    main()

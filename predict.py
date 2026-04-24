import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.cnn import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_samples(num_samples=10):
    # train.py와 동일한 전처리 적용 (학습 때와 입력 형식이 같아야 함)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 테스트 데이터셋에서 샘플 이미지 가져오기 (학습에 사용되지 않은 데이터)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # 저장된 모델 가중치 불러오기
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=DEVICE))
    # map_location: 저장 당시 GPU였더라도 CPU에서 불러올 수 있게 함
    model.eval()  # 평가 모드 (Dropout 비활성화)

    # 2행 5열 격자로 이미지 10개 표시
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()  # 2D 배열 → 1D로 펼쳐서 인덱스로 접근

    with torch.no_grad():  # 예측 시에는 기울기 불필요
        for i in range(num_samples):
            image, label = test_dataset[i]

            # unsqueeze(0): (1, 28, 28) → (1, 1, 28, 28) — 배치 차원 추가 (모델은 배치 단위 입력을 기대함)
            output = model(image.unsqueeze(0).to(DEVICE))
            predicted = torch.argmax(output, dim=1).item()  # 가장 높은 점수의 인덱스 = 예측 숫자

            axes[i].imshow(image.squeeze(), cmap="gray")  # squeeze: 채널 차원 제거 후 흑백으로 표시
            color = "green" if predicted == label else "red"  # 정답이면 초록, 오답이면 빨강
            axes[i].set_title(f"Pred: {predicted} / Label: {label}", color=color)
            axes[i].axis("off")  # 축 눈금 숨기기

    plt.tight_layout()
    plt.savefig("prediction_result.png")  # 이미지 파일로 저장
    plt.show()
    print("결과 이미지 저장: prediction_result.png")


if __name__ == "__main__":
    predict_samples()

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 특징 추출부 (이미지에서 패턴 찾기)
        self.features = nn.Sequential(
            # 1번째 합성곱 레이어: 1채널(흑백) → 32채널
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 → 14x14

            # 2번째 합성곱 레이어: 32채널 → 64채널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
        )

        # 분류부 (추출된 특징으로 숫자 맞추기)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 과적합 방지
            nn.Linear(128, 10),  # 0~9 총 10개 클래스
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

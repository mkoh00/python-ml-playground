import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # ── 특징 추출부 ──────────────────────────────────────────────
        # MNIST(흑백 28×28)와 달리 CIFAR-10은 컬러(RGB 3채널) 32×32 이미지
        # 더 복잡한 이미지이므로 Conv 레이어를 3개로 늘리고 BatchNorm 추가
        self.features = nn.Sequential(

            # [Conv 1] RGB 3채널 → 32채널
            # MNIST는 1채널(흑백)이었지만 여기선 R/G/B 3채널 입력
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 배치 정규화: 레이어 출력을 정규화해 학습 안정화 및 속도 향상
            nn.ReLU(),
            nn.MaxPool2d(2),     # 32×32 → 16×16

            # [Conv 2] 32채널 → 64채널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 16×16 → 8×8

            # [Conv 3] 64채널 → 128채널 (MNIST엔 없던 레이어)
            # CIFAR-10은 고양이/개처럼 복잡한 패턴이 많아 레이어 하나 더 필요
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 8×8 → 4×4
        )

        # ── 분류부 ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 128채널 × 4×4 = 2048개 특징 → 256개 뉴런으로 압축
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),     # MNIST보다 복잡한 데이터라 Dropout 비율을 0.3→0.5로 높임
            nn.Linear(256, 10),  # 10개 클래스: 비행기/자동차/새/고양이/사슴/개/개구리/말/배/트럭
        )

    def forward(self, x):
        x = self.features(x)    # (배치, 3, 32, 32) → (배치, 128, 4, 4)
        x = self.classifier(x)  # (배치, 128, 4, 4) → (배치, 10)
        return x

# python-ml-playground

파이썬 머신러닝 학습 코드를 정리하는 저장소입니다.

---

## 프로젝트 목록

### 📁 mnist_classifier

PyTorch로 구현한 CNN 기반 MNIST 손글씨 숫자 분류기입니다.

**학습 개념**
- CNN 구조 (Conv2d, MaxPool2d, ReLU, Dropout)
- PyTorch 학습 파이프라인 (순전파 → 손실 계산 → 역전파 → 가중치 업데이트)
- 데이터 전처리 및 DataLoader 사용

**구조**
```
mnist_classifier/
├── models/
│   └── cnn.py        # CNN 모델 정의
├── train.py          # 학습 실행
├── predict.py        # 예측 및 시각화
└── requirements.txt
```

**실행 방법**
```bash
pip install -r requirements.txt

# 학습
python train.py

# 예측 (학습 후 실행)
python predict.py
```

**모델 구조**
```
입력 (1×28×28)
  → Conv2d(1→32) + ReLU + MaxPool2d  →  32×14×14
  → Conv2d(32→64) + ReLU + MaxPool2d →  64×7×7
  → Flatten → Linear(3136→128) → Dropout(0.3) → Linear(128→10)
출력 (10개 클래스: 0~9)
```

**학습 환경**
- Python 3.x
- PyTorch
- torchvision

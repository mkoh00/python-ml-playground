import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.cnn import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def predict_samples(num_samples=10):
    # 테스트용 전처리 (증강 없이 정규화만)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("models/cifar10_cnn.pth", map_location=DEVICE))
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_samples):
            image, label = test_dataset[i]
            output = model(image.unsqueeze(0).to(DEVICE))
            predicted = torch.argmax(output, dim=1).item()

            # CIFAR-10은 컬러 이미지(3채널)이므로 표시 방식이 다름
            # (C, H, W) → (H, W, C)로 축 순서 변경 후 정규화 역변환
            img = image.permute(1, 2, 0).numpy()
            img = img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]  # 역정규화
            img = img.clip(0, 1)  # 0~1 범위로 클리핑

            axes[i].imshow(img)
            color = "green" if predicted == label else "red"
            axes[i].set_title(f"Pred: {CLASSES[predicted]}\nLabel: {CLASSES[label]}", color=color)
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()
    print("결과 이미지 저장: prediction_result.png")


if __name__ == "__main__":
    predict_samples()

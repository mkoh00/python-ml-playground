import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.cnn import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_samples(num_samples=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=DEVICE))
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_samples):
            image, label = test_dataset[i]
            output = model(image.unsqueeze(0).to(DEVICE))
            predicted = torch.argmax(output, dim=1).item()

            axes[i].imshow(image.squeeze(), cmap="gray")
            color = "green" if predicted == label else "red"
            axes[i].set_title(f"예측: {predicted} / 정답: {label}", color=color)
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()
    print("결과 이미지 저장: prediction_result.png")


if __name__ == "__main__":
    predict_samples()

import torch
from torchvision import transforms
from PIL import Image
from Assets.MNISTClassifierSurrogate import MNISTClassifierSurrogate

# 加载模型
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifierSurrogate().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, device

# 测试模型准确度
def test_model_accuracy(model, device):
    correct = 0
    total = 0
    testloader = model.testloader
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"模型在测试集上的准确度为: {accuracy:.2f}%")
    return accuracy

# 输入一张图片并预测结果
def predict_image(model, device, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    print(f"预测结果为: {predicted.item()}")
    return predicted.item()

if __name__ == "__main__":
    # 模型路径
    model_path = "./fmnist-alexnet.pt"
    # 测试图片路径
    image_path = "./data/sample_image.png"

    # 加载模型
    model, device = load_model(model_path)

    # 测试模型准确度
    test_model_accuracy(model, device)

    # 输入一张图片并预测结果
    # predict_image(model, device, image_path)

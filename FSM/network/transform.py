
from torchvision import transforms
from PIL import Image
import io
import random

class JPEGCompression(object):
    def __init__(self, quality_range=(55, 90)):
        self.quality_range = quality_range

    def __call__(self, img):
        # 随机生成压缩质量
        quality = random.randint(*self.quality_range)

        # 将图像保存为 JPEG 字节流
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # 重新加载图像
        compressed_img = Image.open(buffer)
        return compressed_img


Data_Transforms = {
    'train': transforms.Compose([

        transforms.RandomHorizontalFlip(),            # Randomly flip images horizontally
        transforms.RandomRotation(10),                # Randomly rotate images within a range of -10 to 10 degrees
        transforms.RandomPerspective(),
        transforms.RandomApply([JPEGCompression()], p=0.5),  # ✅ 使用类代替 Lambda
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # Uncomment these lines if you want to resize or convert images to PIL format
        # transforms.Resize((128, 128)),  # Resize image to 128x128
        # transforms.ToPILImage(),       # Convert to PIL Image

        transforms.ToTensor(),                        # Convert images to tensor format
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize images with mean and std
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # Uncomment these lines if you want to resize or convert images to PIL format
        # transforms.Resize((128, 128)),  # Resize image to 128x128
        # transforms.ToPILImage(),       # Convert to PIL Image

        transforms.ToTensor(),                        # Convert images to tensor format
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize images with mean and std
                             std=[0.229, 0.224, 0.225])
    ]),
}


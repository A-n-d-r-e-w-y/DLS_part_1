import torchvision.models as models
import torch


API_TOKEN = '1827780873:AAFi0Pgiyg3dN0_Rtc6TzILz33CIK9BbJOM'

DIRECTORY_TO_SAVE_IMAGES = "user_images"

AVAILABLE_SIZES = [64, 128, 256, 512]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

MAX_MEMORY_MB = 10.0

DEFAULT_PARAMS = {
            'content_weight': 1.0,
            'style_weight1': 1e4,
            'style_weight2': 1e4,
            'num_steps': 300,
        }

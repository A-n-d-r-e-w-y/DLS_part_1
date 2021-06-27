from PIL import Image
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import resize
import torchvision.transforms as transforms
from __init__ import device, cnn_normalization_mean, cnn_normalization_std, cnn


loader = transforms.Compose([
    transforms.ToTensor()])


def image_loader(image_name, imsize):
    image = Image.open(image_name)
    image_resized = resize(np.array(image), (imsize, imsize), anti_aliasing=False)
    image = loader(image_resized).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, in_data):
        self.loss = F.mse_loss(in_data, self.target)
        return in_data


def gram_matrix(in_data):
    batch_size, f_map_num, h, w = in_data.size()
    features = in_data.view(batch_size * f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
    def __init__(self, target_feature, mode):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.mode = mode
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, in_data):
        if self.mode == 1:
            mask1 = torch.zeros_like(in_data)
            _, _, h, _ = in_data.size()
            mask1[:, :, :h // 2, :] += 1
        else:
            mask1 = torch.zeros_like(in_data)
            _, _, h, _ = in_data.size()
            mask1[:, :, h // 2:, :] += 1
        inpu = in_data.clone()
        inp = inpu * mask1
        G = gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target)
        return in_data


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style1_img, style2_img, content_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    cnn = copy.deepcopy(cnn)
    global mode
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses1 = []
    style_losses2 = []
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            mode = 1

            target_feature1 = model(style1_img).detach()
            style_loss1 = StyleLoss(target_feature1, mode)
            model.add_module("style_loss1_{}".format(i), style_loss1)
            style_losses1.append(style_loss1)
            mode = 2

            target_feature2 = model(style2_img).detach()
            style_loss2 = StyleLoss(target_feature2, mode)
            model.add_module("style_loss2_{}".format(i), style_loss2)
            style_losses2.append(style_loss2)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses1, style_losses2, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       input_img, style1_img, style2_img, content_img, num_steps,
                       style_weight1, style_weight2, content_weight):
    """Run the style transfer."""
    model, style_losses1, style_losses2, content_losses = get_style_model_and_losses(cnn,
                                                                                     normalization_mean,
                                                                                     normalization_std, style1_img,
                                                                                     style2_img, content_img)
    optimizer = get_input_optimizer(input_img)
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score1 = 0
            style_score2 = 0
            content_score = 0
            for sl in style_losses1:
                style_score1 += sl.loss
            for sl in style_losses2:
                style_score2 += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score1 *= style_weight1
            style_score2 *= style_weight2
            content_score *= content_weight
            loss = style_score1 + style_score2 + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss1 : {:4f} Style Loss2 : {:4f} Content Loss: {:4f}'.format(
                    style_score1.item(), style_score2.item(), content_score.item()))
            return style_score1 + style_score2 + content_score

        optimizer.step(closure)
        if run[0] <= num_steps:
            yield "iteration", run[0]

    input_img.data.clamp_(0, 1)
    yield "input_img", input_img


def style_transferring(path_to_content,
                       path_to_style_1,
                       path_to_style_2,
                       imsize,
                       content_weight,
                       style_weight1,
                       style_weight2,
                       num_steps):
    content_img = image_loader(path_to_content, imsize)
    style1_img = image_loader(path_to_style_1, imsize)
    style2_img = image_loader(path_to_style_2, imsize)
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                input_img, style1_img, style2_img, content_img,
                                style_weight1=style_weight1,
                                style_weight2=style_weight2,
                                content_weight=content_weight,
                                num_steps=num_steps)
    return output

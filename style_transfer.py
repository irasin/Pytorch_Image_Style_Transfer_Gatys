import os
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.utils import save_image


def load_image(image_path, transform=None, shape=None, device='cpu'):
    image = Image.open(image_path)

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image.to(device)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        self.slice5 = vgg[21: 30]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5


def calc_content_loss(x, y):
    return F.mse_loss(x, y)


def gram_matrix(x):
    return torch.einsum('bchw, bdhw -> bcd', x, x)


def calc_style_loss(x, y):
    gram_x = gram_matrix(x)
    gram_y = gram_matrix(y)
    return F.mse_loss(gram_x, gram_y)


def calc_tv_loss(img):
    w_variance = F.mse_loss(img[:, :, :, :-1], img[:, :, :, 1:])
    h_variance = F.mse_loss(img[:, :, :-1, :], img[:, :, 1:, :])
    loss = h_variance + w_variance
    return loss


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device='cpu'):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main(config):
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and config.gpu >= 0:
        device = torch.device(f'cuda:{config.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    content = load_image(config.content, transform=trans, device=device)
    style = load_image(config.style, transform=trans, shape=content.shape[-2:], device=device)

    target = content.clone().requires_grad_(True)

    optimizer = LBFGS([target], lr=config.lr)
    vgg = VGG().to(device)

    content_features = vgg(content)
    style_features = vgg(style)

    for i in tqdm.tqdm(range(1, config.iteration + 1)):

        def closure():
            target_features = vgg(target)

            content_loss = calc_content_loss(content_features[3], target_features[3])

            style_loss = 0
            for s, t in zip(style_features, target_features):
                style_loss += calc_style_loss(s, t) * config.style_weight

            tv_loss = calc_tv_loss(target) * config.tv_weight

            loss = style_loss + content_loss + tv_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # print(f'[{i}/{config.iteration}] Content loss: {content_loss.item()}, '
            #       f'Style loss: {style_loss.item()}, Total Variance loss: {tv_loss.item()}')

            return loss

        optimizer.step(closure)

        if i % config.snapshot_interval == 0:
            img = target.detach().to('cpu')
            img = denorm(img)
            save_image(img, f'{config.save_dir}/output_{i}.png')

    c_denorm = denorm(content, device)
    t_denorm = denorm(target, device)
    res = torch.cat([c_denorm, t_denorm], dim=0)
    res = res.to('cpu')

    c_name = os.path.splitext(os.path.basename(config.content))[0]
    s_name = os.path.splitext(os.path.basename(config.style))[0]
    output_name = f'{config.save_dir}/{c_name}_{s_name}'

    save_image(t_denorm, f'{output_name}.jpg', nrow=1)
    save_image(res, f'{output_name}_pair.jpg', nrow=2)

    o = Image.open(f'{output_name}_pair.jpg')
    c = Image.open(config.content)
    s = Image.open(config.style)
    s = s.resize((i // 4 for i in c.size))
    box = (o.width // 2, o.height - s.height)
    o.paste(s, box)
    o.save(f'{output_name}_style_transfer_demo.jpg', quality=95)
    print(f'result saved into files starting with {output_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Style Transfer Using Convolutional Neural Networks,'
                                                 ' CVPR 2016, by Gatys et al.')
    parser.add_argument('--content', '-c',
                        type=str,
                        required=True,
                        help='path to content image')
    parser.add_argument('--style', '-s',
                        type=str,
                        required=True,
                        help='path to style image')
    parser.add_argument('--gpu', '-g',
                        type=int,
                        default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--iteration',
                        type=int,
                        default=100,
                        help='total no. of iterations of the algorithm, default=100')
    parser.add_argument('--snapshot_interval',
                        type=int,
                        default=10,
                        help='interval of snapshot to generate image, default=10')
    parser.add_argument('--style_weight',
                        type=float,
                        default=1000,
                        help='style loss hyperparameter, default=1000')
    parser.add_argument('--tv_weight',
                        type=float,
                        default=0.01,
                        help='total variance loss hyperparameter, default=0.01')
    parser.add_argument('--lr',
                        type=float,
                        default=1,
                        help='learning rate for L-BFGS, default=1')
    parser.add_argument('--save_dir',
                        type=str,
                        default='result',
                        help='save directory for result and loss')
    config = parser.parse_args()
    print(config)
    main(config)

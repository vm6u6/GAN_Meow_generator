import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.nn.init as init



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

class Generator(nn.Module):
    """
    input (N, latent_dim+code_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, classes, channels, img_size, latent_dim, code_dim,dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.img_init_size = self.img_size // 4
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_init_shape = (dim*8, self.img_init_size, self.img_init_size)
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.stem_linear = nn.Sequential(
            nn.Linear(latent_dim + classes + code_dim ,  dim*8*4*4  ,bias=False ),
            nn.BatchNorm1d(dim*8*4*4),
            nn.ReLU())

        self.model = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),           #512x4x4 -> 256x8x8
            dconv_bn_relu(dim * 4, dim * 2),           #256x8x8 -> 128x16x16
            dconv_bn_relu(dim * 2, dim),                  #128x16x16 -> 64x32x32
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),   #3x64x64
            nn.Tanh()
        )
        self.apply(weights_init)


    def forward(self, noise, labels, code):
        z = torch.cat((noise, labels, code), 1)
        z_vec = self.stem_linear(z)
        z_img = z_vec.view(z_vec.shape[0], -1,4,4)
        x = self.model(z_img)
        return x


class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, classes, channels, img_size, latent_dim, code_dim,batch_D=0,dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
                
        self.batch_D = batch_D
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)


        self.model = nn.Sequential(
            nn.Conv2d(self.channels, dim, 5, 2, 2), nn.LeakyReLU(0.2),       # 3x64x64  -> 64x32x32
            conv_bn_lrelu(dim, dim * 2),                                                                    # 64x32x32 ->128x16x16
            conv_bn_lrelu(dim * 2, dim * 4),                                                             # 128x16x16 -> 256x8x8
            conv_bn_lrelu(dim * 4, dim * 8),                                                             # 256x8x8 -> 512 x 4 x 4
        )
        out_linear_dim = dim*8 * (self.img_size // 16) * (self.img_size // 16)

        self.in_features = out_linear_dim
        self.out_features = dim* 2  *(self.img_size // 16) * (self.img_size // 16)
        self.kernel_dims = 16  *(self.img_size // 16) * (self.img_size // 16)
        self.mean = False
        self.T = nn.Parameter(torch.Tensor(self.in_features, self.out_features, self.kernel_dims))
        init.normal_(self.T, 0, 1)

        self.adv_linear =nn.Sequential(
            nn.Conv2d(dim*8, 1, 4),
            nn.Sigmoid())

        self.adv_batch_linear =nn.Sequential(
            #nn.Linear(out_linear_dim + self.out_features, 1),
            nn.Conv2d(dim*10, 1, 4),
            nn.Sigmoid())

             
        self.class_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, self.classes)
        )
        self.code_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.code_dim)
        )

        self.adv_loss = torch.nn.BCELoss( )
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.style_loss = torch.nn.MSELoss()
        self.reg_loss = torch.nn.MSELoss()

    def forward(self, image):
        y_img = self.model(image)
        #y_vec = y_img.view(y_img.shape[0], -1,4,4)
        if (self.batch_D == 1):
            y_vec = y_img.view(y_img.shape[0],-1)
            matrices = y_vec.mm(self.T.view(self.in_features, -1))
            matrices = matrices.view(-1, self.out_features, self.kernel_dims)
            M = matrices.unsqueeze(0)  # 1xNxBxC
            M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
            norm = torch.abs(M - M_T).sum(3)  # NxNxB
            expnorm = torch.exp(-norm)
            o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
            if self.mean:
                o_b /= y_vec.size(0) - 1

            y = torch.cat([y_vec, o_b], 1)
            y = y.view(y.shape[0],-1,4,4)
            y = self.adv_batch_linear(y)
        else:
            y_vec = y_img.view(y_img.shape[0], -1,4,4)
            y = self.adv_linear(y_vec)

        y_vec = y_vec.view(y_img.shape[0],-1)
        label = F.softmax(self.class_linear(y_vec), dim=1)
        code = self.code_linear(y_vec)
        return y.view(-1), label, code,y_vec
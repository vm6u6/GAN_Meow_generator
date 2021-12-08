import itertools
from torch.utils.data import Dataset, DataLoader
import cv2,os,glob,csv
import torchvision.transforms as transforms
import random
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import argparse

from cat_dataset import CatDataset
from nets.InfoGan import Generator as infoganG
from nets.InfoGan import Discriminator as infoganD





def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def _to_onehot(var, dim):
    res = torch.zeros((var.shape[0], dim)).cuda()
    res[range(var.shape[0]), var] = 1.
    res *= 0.9
    res += 0.1/res.shape[1]
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument_group('Dataset related arguments')
    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Data Directory')
    parser.add_argument('--dataset', type=str, default="flowers",
                        help='Dataset to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Total number of epochs to train')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume epoch to resume training')
    parser.add_argument('--save_dir', type=str, default='./sample')
    parser.add_argument('--workspace_dir', type=str, default='./models')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--style_dim', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=2*1e-4)
    parser.add_argument('--l1_p', type=float, default=10)
    parser.add_argument('--epoch_decay', type=float, default=100,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--batch_D', type=bool, default=False)



    args = parser.parse_args()

    dataset = CatDataset(args.dataset,args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    classes = dataset.num_class()
    netG = infoganG(classes , args.channels, args.img_size, args.latent_dim, args.style_dim)
    netG.apply(_weights_init)
    netG.cuda()

    netD = infoganD(classes, args.channels, args.img_size, args.latent_dim, args.style_dim,batch_D=args.batch_D)
    netD.apply(_weights_init)
    netD.cuda()

    g_lr = args.g_lr
    d_lr = args.d_lr

    if args.resume_epoch:
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(args.resume_epoch))
        G_path = os.path.join(args.workspace_dir, 'G_epoch{}.pth'.format(args.resume_epoch))
        D_path = os.path.join(args.workspace_dir, 'D_epoch{}.pth'.format(args.resume_epoch))
        netG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        netD.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        start_epoch = args.resume_epoch
        d_lr /= 2 ** (args.resume_epoch // args.epoch_decay)
        g_lr /= 2 ** (args.resume_epoch // args.epoch_decay) 
        resume_epoch = args.resume_epoch
    else:
        resume_epoch = 1





    optim_G = optim.Adam(netG.parameters(),lr=args.g_lr,betas=(0.5, 0.999))
    optim_D = optim.Adam(netD.parameters(),lr=args.d_lr,betas=(0.5, 0.999))
    optim_info = torch.optim.Adam(itertools.chain(netG.parameters(), netD.parameters()),lr=args.d_lr, betas=(0.5, 0.999))
    n_viz = 36
    viz_noise = torch.randn(n_viz , args.latent_dim).uniform_(-1,1).cuda()
    nrows =n_viz  // 6
    viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(6)])).cuda()
    
    viz_onehot = _to_onehot(viz_label, dim=classes)
    viz_style = torch.zeros((n_viz , args.style_dim)).cuda()
    G_L_list = []
    D_L_list = []
    Info_list = []
    for epoch in range(resume_epoch,args.num_epochs+1):
        G_L = 0
        D_L = 0
        Info_L = 0
        for batch_idx, sample  in enumerate(data_loader):
            data = sample['true_imgs'].cuda()
            target = sample['true_embed'].cuda()
            batch_size = data.shape[0]

            real_label = torch.ones([data.shape[0],1]).cuda()
            fake_label = torch.zeros([data.shape[0],1]).cuda()

            z_noise = torch.randn(batch_size, args.latent_dim).uniform_(-1.0,1.0).cuda()
            #z_noise.data.uniform_(-1.0,1.0)
            x_fake_labels = torch.randint(0, classes, (batch_size,)).cuda()
            labels_onehot = _to_onehot(x_fake_labels, dim=classes)
            z_style = torch.zeros((batch_size, args.style_dim)).uniform_(-1.0,1.0).cuda()
            x_fake = netG(z_noise, labels_onehot, z_style)

            # Train D
            netD.zero_grad()
            y_real, label_true, _,y_real_vec = netD(data)
            d_real_loss = netD.adv_loss(y_real, real_label)

            y_fake_d, _, _ ,_= netD(x_fake.detach())
            d_fake_loss = netD.adv_loss(y_fake_d, fake_label)
            
            #d_vector_loss = netD.reg_loss(y_fake_vec,y_real_vec)
            d_loss = (d_real_loss + d_fake_loss + args.l1_p*netD.class_loss(label_true,target)) / (2+args.l1_p)
            d_loss.backward()
            optim_D.step()



            # Train G and Q
            netG.zero_grad()

            x_fake = netG(z_noise, labels_onehot, z_style)
            y_fake_g,label_fake, style_fake,_ = netD(x_fake)

            #y_real, _, _,y_real_vec = netD(data)
            #g_reg_loss = netD.reg_loss(x_fake,data)
           # g_loss = netD.adv_loss(y_fake_g, real_label) + args.l1_p * g_reg_loss
            g_loss =  (netD.adv_loss(y_fake_g, real_label)\
                                +args.l1_p *netD.class_loss(label_fake, x_fake_labels)\
                                +args.l1_p * netD.style_loss(style_fake, z_style))/(1+2*args.l1_p)
            g_loss.backward()
            optim_G.step()


            '''
            # Update mutual information
            optim_info.zero_grad()
            z_noise.normal_()
            x_fake_labels = torch.randint(0, classes, (batch_size,)).cuda()
            labels_onehot = _to_onehot(x_fake_labels, dim=classes)
            z_style.normal_()
            x_fake = netG(z_noise, labels_onehot, z_style)
            _, label_fake, style_fake,_ = netD(x_fake)
            _,label_true,_,_ = netD(data)
            
            #info_loss = 10 * netD.class_loss(label_fake, x_fake_labels) +  netD.style_loss(style_fake, z_style)
            info_loss =  0.4*netD.class_loss(label_true,target)\
                                    +  0.4 * netD.class_loss(label_fake, x_fake_labels) \
                                    +  0.2 * netD.style_loss(style_fake, z_style)
            info_loss.backward()
            optim_info.step()
            '''
            G_L += g_loss.item()/len(data_loader)
            D_L += d_loss.item()/len(data_loader)
            #Info_L += info_loss.item()/len(data_loader)

            # log
            print(f'\rEpoch [{epoch}/{args.num_epochs}] {batch_idx+1}/{len(data_loader)} ', end='')
        #print(f'\rEpoch [{epoch}/{args.num_epochs}] {batch_idx+1}/{len(data_loader)} G_Loss: {G_L:.4f} D_Loss: {D_L:.4f} Info_Loss: {Info_L:.4f}', end='')
        print(f'\rEpoch [{epoch}/{args.num_epochs}] {batch_idx+1}/{len(data_loader)} G_Loss: {G_L:.4f} D_Loss: {D_L:.4f} ', end='')

        G_L_list .append(G_L)
        D_L_list.append(D_L)
        Info_list.append(0)
        #Info_list.append(Info_L)

        netG.eval()
        f_imgs_sample = (netG(viz_noise, viz_onehot, viz_style).data+1)/2.0
        os.makedirs(args.save_dir, exist_ok=True)
        filename = os.path.join(args.save_dir, f'Epoch_{epoch:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=6)
        print(f' | Save some samples to {filename}.')
        netG.train()
        os.makedirs(args.workspace_dir, exist_ok=True)
        with open(os.path.join(args.workspace_dir, 'training_loss.csv'),'w',newline='') as csvfile:
            c = csv.writer(csvfile)
            c.writerow(['Generator_loss','Discriminator_loss','Information_loss'])
            for i in range(len(G_L_list)):
                c.writerow([G_L_list[i],D_L_list[i],Info_list[i]])
        if (epoch) % 50 == 0:
            torch.save(netG.state_dict(), os.path.join(args.workspace_dir, 'G_epoch%d.pth'%epoch))
            torch.save(netD.state_dict(), os.path.join(args.workspace_dir, 'D_epoch%d.pth'%epoch))

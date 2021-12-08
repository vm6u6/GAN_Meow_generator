import metrics
 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='folder' ,help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot_real', default='results/real/0', help='path to dataset real')
parser.add_argument('--dataroot_fake', default='.results/fake/0', help='path to dataset fake')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
 
opt = parser.parse_args()
 
#inception_v3
s = metrics.compute_score_raw(opt.dataset, opt.imageSize, opt.dataroot_real,  opt.batchSize, opt.outf+'/real', opt.outf+'/fake',
                                 opt.dataroot_fake, conv_model='inception_v3', workers=int(opt.workers))
for i in range(len(s)):
    print(i,"=", s[i])

import os
import sys
import torch
import torch
import torch.nn as nn
import numpy as np
import argparse
import surfa as sf
import scipy.ndimage
from models.model import UNet
from models.utils import normalize

description = ''' 
Neural Pre-processing (NPP) converts Head MRI images
to an intensity-normalized, skull-stripped brain in a standard coordi-
nate space. If you use NPP in your analysis, please cite:
'''

# os.system(f'python npp.py -i {inputfname} -o {outputdir}')

# parse command line
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--image', metavar='file', required=True, help='Input image to pre-processing.')
parser.add_argument('-o', '--out', metavar='file', help='Save stripped image to path.')
parser.add_argument('-w', '--weight', metavar='float', help='Smoothness of intensity normalization mapping. The range of smoothness is [-3,2],'
                                                            ' where a larger value implies a higher degree of smoothing',default =-1)
parser.add_argument('-g', '--gpu', action='store_true', help='Use the GPU.')

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

# sanity check on the inputs
if not args.out:
    sf.system.fatal('Must provide at least --out output flags.')
elif not os.path.exists(os.path.dirname(args.out)):
    sf.system.fatal('Output directory does not exist.')

# check args.weight is in the range and float
if args.weight:
    args.weight = float(args.weight)
    if args.weight < -3 or args.weight > 2:
        sf.system.fatal('The range of smoothness should within [-3,2], where a larger value implies a higher degree of smoothing')

# necessary for speed gains (I think)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# configure GPU device
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    device_name = 'GPU'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')
    device_name = 'CPU'

# configure model
print(f'Configuring model on the {device_name}')


with torch.no_grad():
    model = UNet()
    model.to(device)
    model.eval()


version = '1'
print(f'Running Neural Pre-processing model version {version}')
cwd = os.getcwd()
modelfile = os.path.join(cwd,'models/checkpoints', f'npp_v{version}.pth')
checkpoint = torch.load(modelfile, map_location=device)

model.load_state_dict(checkpoint)

# load input volume
image = sf.load_volume(args.image)
print(f'Input image read from: {args.image}')

# frame check
if image.nframes > 1:
    sf.system.fatal('Input image cannot have more than 1 frame')

#i normalize image to [0, 255] and to [0, 1]
image = normalize(image)

# conform image and fit to shape with factors of 64
conformed = image.conform(voxsize=1.0, dtype='float32',shape=(256,256,256), method='nearest', orientation='LIA')

# predict the surface distance transform
with torch.no_grad():
    input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
    output = model(input_tensor,args.weight)
    mni_norm = output[0].cpu().numpy().squeeze().astype(np.int16)
    norm = output[1].cpu().numpy().squeeze().astype(np.int16)
    scalar_field = output[2].cpu().numpy().squeeze().astype(np.int16)

# unconform the sdt and extract mask
mni_norm = conformed.new(mni_norm)#.resample_like(image,method='nearest', fill=0)
norm = conformed.new(norm)#.resample_like(image, method='nearest',fill=0)
scalar_field = conformed.new(scalar_field)#.resample_like(image, method='nearest',fill=0)

# write the masked output
if args.out:
    filename = os.path.basename(args.image)
    filename = filename.split('.')[0]
    mni_norm.save(os.path.join(args.out,filename+'_mni_norm.nii.gz'))
    norm.save(os.path.join(args.out,filename+'_norm.nii.gz'))
    scalar_field.save(os.path.join(args.out,filename+'_scalar_field.nii.gz'))

    print(f'Results saved to: {args.out}')

print('If you use Neural Pre-processing in your analysis, please cite:')

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


def get_affine_from_model(model, input_tesnor):
    from einops import rearrange
    input_downsampled = torch.nn.functional.interpolate(input_tesnor, size=[128, 128, 128],
                                                        mode='trilinear', align_corners=False)
    x0_0 = model.conv0_0(input_downsampled)
    x1_0 = model.conv1_0(x0_0)
    x2_0 = model.conv2_0(x1_0)
    x3_0 = model.conv3_0(x2_0)
    x4_0 = model.conv4_0(x3_0)
    x5_0 = model.conv5_0(x4_0)
    identity = torch.eye(3, 4).repeat(x5_0.shape[0], 1, 1).type_as(x5_0)
    b, c, h, w, d = x5_0.shape
    x5_0_faltten = rearrange(x5_0, 'b c h w d-> b (h w d) c')
    x5_0_faltten = model.attention1(model.LN1(x5_0_faltten)) + x5_0_faltten
    x5_0_faltten = model.mlp1(model.LN2(x5_0_faltten)) + x5_0_faltten
    x5_0_faltten = model.attention2(model.LN3(x5_0_faltten)) + x5_0_faltten
    x5_0_faltten = model.mlp2(model.LN4(x5_0_faltten)) + x5_0_faltten
    affine = model.head(x5_0_faltten.mean(dim=1)).reshape(-1, 3, 4) + identity
    return affine

def apply_affine_image(affine, input_tensor):
    x0_0_warp = torch.nn.functional.affine_grid(affine, input_tensor.size(), align_corners=False)
    # norm = input_tensor * 255 * output_upsampled
    # mni_norm = torch.nn.functional.grid_sample(norm, x0_0_warp, align_corners=False)
    output = torch.nn.functional.grid_sample(input_tensor*255, x0_0_warp, align_corners=False)
    return output

def apply_affine_label(affine, input_tensor):
    x0_0_warp = torch.nn.functional.affine_grid(affine, input_tensor.size(), align_corners=False)
    # norm = input_tensor * 255 * output_upsampled
    # mni_norm = torch.nn.functional.grid_sample(norm, x0_0_warp, align_corners=False)
    output = torch.nn.functional.grid_sample(input_tensor, x0_0_warp, align_corners=False, mode='nearest')
    return output


description = ''' 
This script moves segmentation outputs from FastSurfer to NPP processed image in MNI space
'''

# os.system(f'python nppy_fastsurfer_segmentation2mni.py -i {inputfname} -l {labelfname} -o {outputdir}')

def main():
    # parse command line
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--image', metavar='file', required=True, help='Input image used for FastSurfer and/or NPP')
    parser.add_argument('-l', '--label', metavar='file', required=True, help='Input moving label')
    parser.add_argument('-o', '--out', metavar='file', help='Save stripped image to path.')
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

    inputfname = args.image
    labelfname = args.label

    image = sf.load_volume(inputfname)
    label = sf.load_volume(labelfname)

    # frame check
    if image.nframes > 1:
        sf.system.fatal('Input image cannot have more than 1 frame')

    #i normalize image to [0, 255] and to [0, 1]
    image = normalize(image)
    # label = normalize(label)

    # conform image and fit to shape with factors of 64
    conformed = image.conform(voxsize=1.0, dtype='float32',shape=(256,256,256), method='nearest', orientation='LIA')
    conformed_label = label.conform(voxsize=1.0, dtype='float32',shape=(256,256,256), method='nearest', orientation='LIA')

    # predict transform
    with torch.no_grad():
        input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
        label_tensor = torch.from_numpy(conformed_label.data[np.newaxis, np.newaxis]).to(device)
        affine2mni = get_affine_from_model(model, input_tensor)
        transformed_label = apply_affine_label(affine2mni, label_tensor)
        # transformed_image = apply_affine_image(affine2mni, input_tensor)

    transformed_label = (transformed_label).detach().cpu().numpy().squeeze().astype(np.int16)
    # transformed_image = (transformed_image).detach().cpu().numpy().squeeze().astype(np.int16)

    transformed_label = conformed.new(transformed_label)
    # transformed_image = conformed.new(transformed_image)

    if args.out:
        filename = os.path.basename(args.image).split('.nii')[0]
        labelfname = os.path.basename(args.label).split('.mgz')[0]
        transformed_label.save(os.path.join(outputdir, filename + '_' + labelfname + '_mni.nii.gz'))
        print(f'Results saved to: {args.out}')


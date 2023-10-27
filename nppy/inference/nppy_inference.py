import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import numpy as np
import argparse
import surfa as sf
from models.model import UNet
from models.utils import normalize,find_dicom_folders,find_nii_files,load_dicom_series
import nibabel as nib
from huggingface_hub import hf_hub_download
from surfa import Volume

description = ''' 
Neural Pre-processing (NPP) converts Head MRI images
to an intensity-normalized, skull-stripped brain in a standard coordi-
nate space. If you use NPP in your analysis, please cite:
'''

def main():
    # parse command line
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', metavar='folder_path', required=True, help='Input folder to pre-processing.')
    parser.add_argument('-o', '--output_folder', metavar='folder_path', help='Save stripped image to the output folder.')
    parser.add_argument('-w', '--weight', metavar='float', help='Smoothness of intensity normalization mapping. The range of smoothness is [-3,2],'
                                                                ' where a larger value implies a higher degree of smoothing',default =-1)
    parser.add_argument('-s', '--field', action='store_true', help='Save the scalar field map.')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use the GPU.')

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()

    # args.input_folder = '/Users/hexinzi/Downloads/input'
    # args.output_folder = '/Users/hexinzi/Downloads/output'
    # args.weight = -1
    # args.gpu = True
    # args.field = False
    # download the model
    #project_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = hf_hub_download(repo_id="hexinzi/NeuralPreProcessing", filename="npp_v1.pth")

    #if input_folder is file, convert to folder
    if os.path.isfile(args.input_folder):
        args.input_folder = os.path.dirname(args.input_folder)

    # if input and output are same, exit
    if args.input_folder == args.output_folder:
        sf.system.fatal('Input and output folders are same. Exiting.')

    # find all dicom folders and nii files, generate output file path and print it
    dicom_folders_path = find_dicom_folders(args.input_folder)
    nii_files_path = find_nii_files(args.input_folder)
    input_image_path = dicom_folders_path + nii_files_path
    relative_input_image_path = [os.path.splitext(os.path.relpath(i,args.input_folder))[0] for i in input_image_path]
    out_file_path_without_extension = [os.path.join(args.output_folder, i) if input_image_path[ind] in nii_files_path else os.path.join(args.output_folder, i,'dicom') for ind,i in enumerate (relative_input_image_path)  ]

    print(f'input_folder path:\n',"\n".join(relative_input_image_path))

    # sanity check on the inputs
    if not args.output_folder:
        sf.system.fatal('Must provide at least --output_folder output flags.')
    elif not os.path.exists(os.path.dirname(args.output_folder)):
        print('Output directory does not exist. Create output directory.')
        os.makedirs(os.path.dirname(args.output_folder),exist_ok=True)

    # check args.weight is in the range and float
    if args.weight:
        args.weight = float(args.weight)
        if args.weight < -3 or args.weight > 2:
            sf.system.fatal('The range of smoothness should within [-3,2], where a larger value implies a higher degree of smoothing')

    # necessary for speed gains (I think)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # configure GPU device

    is_mps_available = torch.backends.mps.is_available()
    is_gpu_available = torch.cuda.is_available()

    if args.gpu and is_gpu_available:
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


    version = '0.1'
    print(f'Running Neural Pre-processing model version {version}')
    #cwd = os.getcwd()
    #modelfile = os.path.join(cwd,'models/checkpoints', f'npp_v{version}.pth')
    modelfile = checkpoint_path
    checkpoint = torch.load(modelfile, map_location=device)

    model.load_state_dict(checkpoint)


    for input_path, output_path in zip(input_image_path,out_file_path_without_extension):
        if input_path in dicom_folders_path:
            # load input volume
            nib_image = load_dicom_series(input_path)
            data = np.asanyarray(nib_image.dataobj)
            #np.clip(data, -10, 200, out=data)
            image = sf.io.framed.framed_array_from_4d(data=data, atype = Volume)
            voxsize = nib_image.header['pixdim'][1:4]
            image.geom.update(vox2world=nib_image.affine, voxsize=voxsize)
        else:
        # load input volume
            image = sf.load_volume(input_path)
        print(f'Input image read from: {input_path}')

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
            scalar_field = output[2].cpu().numpy().squeeze().astype(np.float32)

        # unconform the sdt and extract mask
        mni_affine = np.array([[-1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
        mni_norm_nib_handler = nib.Nifti1Image(mni_norm, affine=mni_affine)
        conformed.data = (conformed.data*255).astype(np.int16)
        norm = conformed.new(norm) #.resample_like(image, method='nearest',fill=0)
        scalar_field = conformed.new(scalar_field) #.resample_like(image, method='nearest',fill=0)

        # write the masked output
        if output_path:
            os.makedirs(os.path.dirname(output_path),exist_ok=True)
            nib.save(mni_norm_nib_handler, output_path+'_mni_norm.nii.gz')
            conformed.save(output_path+'_orig.nii.gz')
            norm.save(output_path+'_norm.nii.gz')
            if args.field:
                scalar_field.save(output_path+'_scalar_field.nii.gz')
            print(f'Results saved to: {args.output_folder}')

    print('If you use Neural Pre-processing in your analysis and find it useful, please cite: \n'
          'He, X., Wang, A.Q., Sabuncu, M.R. (2023). Neural Pre-processing: A Learning Framework for End-to-End Brain MRI Pre-processing. \n'
          'In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. \n'
          'Lecture Notes in Computer Science, vol 14227. Springer, Cham. https://doi.org/10.1007/978-3-031-43993-3_25')

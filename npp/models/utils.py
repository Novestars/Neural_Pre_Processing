# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pydicom
import SimpleITK as sitk
import nibabel as nib
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def quantize_ste(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x


def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)




def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.
    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    #print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    #print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale



def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    #print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def normalize(img):
    src_min, scale = getscale(img.data, 0, 255)

    if not img.data.dtype == np.dtype(np.uint8):
        new_data = scalecrop(img.data, 0, 255, src_min, scale)
    else:
        new_data = img.data

    img.data = new_data/255

    return img

def find_nii_files(root_folder):
    """
    Walk through root_folder to find all .nii and .nii.gz files.

    Parameters:
    - root_folder: The starting directory from which to begin the search.

    Returns:
    - A list of full paths to .nii and .nii.gz files.
    """
    nii_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                full_path = os.path.join(dirpath, filename)
                nii_files.append(full_path)

    return nii_files

def is_dicom_file(filepath):
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False

def find_dicom_folders(root_folder):
    """
    Walk through root_folder to find all folders containing DICOM files.

    Parameters:
    - root_folder: The starting directory from which to begin the search.

    Returns:
    - A set of directories containing at least one DICOM file.
    """
    dicom_folders = set()

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if is_dicom_file(full_path):
                dicom_folders.add(dirpath)
                break  # Once a DICOM file is found in a folder, no need to check other files in the same folder

    return list(dicom_folders)


def make_affine(simpleITKImage):
    # get affine transform in LPS
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine

def load_dicom_series(directory):
    """Load DICOM series from a directory."""
    # Get the list of DICOM series IDs in the directory
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)

    if not series_ids:
        raise ValueError(f"No DICOM series found in {directory}")

    # For simplicity, load the first series.
    # Modify if you have more than one series and you want to load them differently.
    dicom_series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series_ids[0])

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_series_file_names)
    img = reader.Execute()
    img = sitk.PermuteAxes(img, [0, 2, 1])

    affine = make_affine(img)
    img = nib.Nifti1Image( sitk.GetArrayFromImage(img).transpose(), affine)

    return img



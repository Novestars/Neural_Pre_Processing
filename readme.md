# Neural Pre-processing Python(NPPY) 
![Top: An overview of Neural Pre-processing. Bottom: The network archiecture of Neural Pre-processing](figure/pipeline_v2.png)

**Objective**: NPPY is an end-to-end weakly supervised learning approach for converting raw head MRI images to intensity-normalized, skull-stripped brain in a standard coordinate space. 

**Methods**: NPPY solves three sub-tasks simultaneously through a neural network, without individual sub-task supervision. The sub-tasks include geometric-preserving intensity mapping, spatial transformation, and skull stripping. The model disentangles intensity mapping and spatial normalization to solve the under-constrained objective. 

**Results**: NPPY outperforms the state-of-the-art methods, which tackle only a single sub-task, according to quantitative results. The importance of NPP's architecture design is demonstrated through ablation experiments. Additionally, NPP provides users with the flexibility to control each task during inference time. 

# Instructions
Neural Pre-processing Python(NPPY) can be accessed as a python library package.
 
## Python library package
To use the NPP python (nppy) library, either clone this repository and install via 'pip install -e .' or install directly with pip.

```
pip install nppy
```

### Usage
Once you download the Neural Pre-processing docker script, you can use NPP with the following command-line syntax:
 

``` 
nppy -i input_folder -o output_folder -s -g -w -1
```

In this command, "input_folder" represents the path to the input folder and "output_folder" is the pre-processed output folder path. NPP generates brain mask, intensity normalized brain and intensity normalized brain in the standard coordinate space. 
For the large majority of images with voxel sizes near 1 mm3, NPP should run in less than 10 seconds on the CPU. As image size or resolution increases, the runtime might increase as well. If you encounter any issues, please contact the NPP development team for support.

# NPPY Papers

If you use NPPY or some part of the code, please cite:
 
  * Neural Pre-processing:   

    **Neural Pre-Processing: A Learning Framework for End-to-end Brain MRI Pre-processing.**  
[Xinzi He](https://www.bme.cornell.edu/research/grad-students/xinzi-he), Alan Wang, [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
Medical Image Computing and Computer Assisted Intervention 2023. \
[Springer](https://doi.org/10.1007/978-3-031-43993-3_25)\
[arXiv:2303.12148](https://arxiv.org/abs/2303.12148)

# Pre-trained models
See list of pre-trained models available [here](https://www.dropbox.com/s/zbwuqinhuvf0thz/npp_v1.pth?dl=0).

# Data:
The Neural Pre-processing python(NPPY) model has been trained on a combination of seven different datasets, namely GSP, ADNI, OASIS, ADHD, ABIDE, MCIC, and COBRE. However, please note that most of the data used in the NPP papers cannot be shared due to redistribution restrictions. Nonetheless, if any researcher requires access to the Freesurfer outputs utilized during the training process, please feel free to contact me.



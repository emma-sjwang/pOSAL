# *p*OSAL: Patch-based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation.


-------

We provide the Keras implements based on Tensorflow Backend for REFUGE challenge segmentation task.
<img src="https://github.com/EmmaW8/pOSAL/blob/master/imgs/overview.png" width="800px"/>

      
## Getting Started

### Install requirments
``` bash
conda create -n posal python=3.5
conda activate posal
pip install keras==2.2.0
pip insatll tensorflow-gpu==1.4.0
conda install tqdm
conda install -c anaconda scikit-image
conda install opencv
```
### Prerequisites

- GPU, CUDA=9.0


### Running Evaluation

- Clone this repo:
```bash
git clone https://github.com/EmmaW8/pOSAL.git
cd pOSAL
```

### To reproduce the results for the rank in REFUGE challenge in MICCAI 2018, please do
``` bash
python predict.py 0 # 0 is the avaliable GPU id, change is neccesary

```

### Running Training for Dri-GS dataset

Remember to check/change the data path and weight path

```bash
python train_DGS.py 0
python test_DGS.py 0
```

### The CDR values used for glaucoma diagnsis are generated with MATLAB.
```
cd matlab-code
```

Please change the input and output path in the `generate_CDR_values.m` file.



**Acknowledge**
Some codes are revised according to selimsef/dsb2018_topcoders, [HzFu/MNet_DeepCDR](https://github.com/HzFu/MNet_DeepCDR) and [evaluateion code ](https://github.com/ignaciorlando/refuge-evaluation).
Thank them very much.
 

### Citation
```
@article{wang2019patch,
  journal={IEEE Transactions on Medical Imaging},
  title={Patch-Based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation},
  author={Wang, Shujun and Yu, Lequan and Yang, Xin and Fu, Chi-Wing and Heng, Pheng-Ann},
  year={2019},
  volume={38},
  number={11},
  pages={2485-2495},
  publisher={IEEE},
  doi={10.1109/TMI.2019.2899910},
  }
```  

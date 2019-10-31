# *p*OSAL: Patch-based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation.


-------

We provide the Keras implements based on Tensorflow Backend for REFUGE challenge segmentation task.
<img src="https://github.com/EmmaW8/pOSAL/blob/master/imgs/overview.png" width="800px"/>

      
## Getting Started
### Prerequisites

- python 3.5
- tensorflow 1.4.0
- keras 2.2.0
- GPU, CUDA

### Packages

- tqdm
- skimage
- opencv
- scipy
- matplotlib


### Running Evaluation

- Clone this repo:
```bash
git clone https://github.com/EmmaW8/pOSAL.git
cd pOSAL
python predict.py
```

### Running Training for Dri-GS dataset
```bash
python train_DGS.py
python test_DGS.py
```

Before running test, please check whether the model weight path is correct.




**Acknowledge**
Some codes are revised according to selimsef/dsb2018_topcoders and HzFu/MNet_DeepCDR.
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

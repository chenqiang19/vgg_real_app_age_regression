# On the effect of age perception biases for real age regression

This is implementation of the paper **"On the effect of age perception biases for real age regression"** for age regression, accepted in the 14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019).

# Environment
1. python 3.7
2. pytorch 1.3.2
3. CUDA 10.1
4. cudnn 7.6.5
5. Linux Ubuntu 18.04 LTS
  ```Shell
  git clone https://github.com/qiangchen19/vgg_real_app.git
  ```

# Data Setup
1. Download the preprocessed data ([train + valid + test set, + pre-trained model](https://drive.google.com/file/d/1KF_eq_-1uv1zHCCpiNTM00h3APuShLbf/view?usp=sharing)).

2. Create Directory 
  ```Shell
  mkdir data/data_h5
  mkdir data/data_h5
  ```
# Intructions

This code is support for pytorch. 

Running the code (training and predicting).

Parameters are defined as: [data_path, train_model (True|False),  stage_num (1|2), lr (current), batch_size, epochs, lr (stage 1), optims(Adam), use_gpu].

Code test only in gpu.

# Citation

1. The reference paper ([arXiv link](https://arxiv.org/abs/1902.07653)) as:
2. @inproceedings{jacques:FG2019,
    author={Julio C. S. Jacques Junior and Cagri Ozcinar and Marina Marjanovic and Xavier Baro and Gholamreza Anbarjafari and Sergio Escalera},
    booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
    title={On the effect of age perception biases for real age regression},
    year={2019},
    }
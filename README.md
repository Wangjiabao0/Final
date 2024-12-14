[!IMPORTANT]
Algorithm does not get a good performance, further debug or tuning is needed.

# Overview
This project implements some exploration methods in reinforcement learning, like **Noisy Networks**, **State Entropy Maximization**, **curiosity-driven** and **Dropout**.

# Dependencies
Ensure you have the following libraries installed:
- gymnasium = 0.29.1
- numpy = 1.22.0
- torch = 1.13.0+cu116
- torchaudio = 0.13.0+cu116
- torchvision = 0.14.0+cu116
- tensorboard = 2.14.0
- tensorboard-data-server = 0.7.2
- moviepy = 1.0.3


Install the dependencies with:
```bash
pip install -r requirements.txt
```

# How to Run
## Training
Run the training script using the following command:
```bash
python train.py --config ./cfg/train.yaml
```
## Testing
Run the testing script using the following command:
```bash
python test.py --config ./cfg/test.yaml
```
One can choose if save video or not by setting geh yaml file. The save folder is {exp_path}/video

## Draw figures
eg. run the drawer script using the following command:
```bash
python utils/draw_reward.py --exp ./exp/A2C_241211_185757/
```
The result will save in folder ./exp/A2C_241211_185757/

## Configuration
- The --config flag specifies the path to the configuration file.
- Modify the YAML files in ./cfg/ to adjust hyperparameters or environment settings.

# Reference
[1] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, “Curiosity-driven exploration by self-supervised prediction,” in International conference on machine learning. PMLR, 2017. \
[2] M. Fortunato, M. G. Azar, B. Piot, J. Menick, M. Hessel, I. Osband, A. Graves, V. Mnih, R. Munos, D. Hassabis, et al., “Noisy networks for exploration,” in International Conference on Learning Representations, 2018.\
[3] T. T. Sung, D. Kim, S. J. Park, and C.-B. Sohn, “Dropout acts as auxiliary exploration,” International Journal of Applied Engineering Research, vol. 13, no. 10, pp. 7977–7982, 2018.\
[4] Y. Seo, L. Chen, J. Shin, H. Lee, P. Abbeel, and K. Lee, “State entropy maximization with random encoders for efficient exploration,” in International Conference on Machine Learning. PMLR, 2021.\
[5] Yuan, M., et al. "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning," in arXiv preprint arXiv:2405.19548, 2024.

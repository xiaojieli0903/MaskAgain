# Official PyTorch Implementation of MMKD.

![MMKD Framework](figs/mmkd.jpg)




> [**Mask Again: Masked Knowledge Distillation for Masked Video Modeling**]
<!-- (https://arxiv.org/abs/)<br> -->
> [Xiaojie Li](https://github.com/xiaojieli0903), [Shaowei He], [Jianlong Wu], [Yue Yu], [Liqiang Nie], [Min Zhang]<br>Harbin Institute of Technology

<!-- ## 📰 News
**[2023.5.20]** The pre-trained models and scripts of **ViT-S** and **ViT-B** are available! <br> -->
<!-- **[2023.6.15]** MMKD is accepted by **ACMMM 2023**! 🎉 <br> -->


## 🚀 Main Results

### ✨ Kinetics-400

|  Method  | Extra Data | Backbone | Resolution | #Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-S   |  224x224   |         16x5x3          | 78.7  | 93.6  |
| VideoMAE |  ***no***  |  ViT-B   |  224x224   |         16x5x3          | 81.0  | 94.6  |



### ✨ UCF101 & HMDB51

|  Method  |  Extra Data  | Backbone | UCF101 | HMDB51 |
| :------: | :----------: | :------: | :----: | :----: |
| VideoMAE | Kinetics-400 |  ViT-S   |  92.9  |  72.0  |
| VideoMAE | Kinetics-400 |  ViT-B   |  96.2  |  77.1 |

## 🔨 Installation

Please follow the instructions in [INSTALL.md](INSTALL.md).

## 📍Model Zoo

We provide pre-trained and fine-tuned models in [MODEL_ZOO.md](MODEL_ZOO.md).

## 👀 Visualization

We provide the script for visualization in [`vis_kd.sh`](scripts/vis_kd.sh).

## 👍 Acknowledgements

This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE.git). Thanks to the contributors of this great codebase.


## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{li2023mask,
  title={Mask Again: Masked Knowledge Distillation for Masked Video Modeling},
  author={Li, Xiaojie and He, Shaowei and Wu, Jianlong and Yu, Yue and Nie, Liqiang and Zhang, Min},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2221--2232},
  year={2023}
}
```

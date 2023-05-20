# MMKD Model Zoo

### Kinetics-400

|  Method  | Extra Data | Backbone | Pre-train Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| MMKD |  ***no***  |  ViT-S   |  400  | 16x5x3  | [script](scripts/kd_small_k400_slurm.sh)/[log](https:)/[checkpoint](https:) | [script](scripts/finetune_small_k400_slurm.sh)/[log](https:)/[checkpoint](https:) | 78.7 | 93.6 |
| MMKD |  ***no***  |  ViT-B   |  400  | 16x5x3  | [script](scripts/kd_base_k400_slurm.sh)/[log](https:)/[checkpoint](https:) | [script](scripts/finetune_base_k400_slurm.sh)/[log](https:)/[checkpoint](https://) | 81.0  | 94.6  |




### UCF101

|  Method  | Extra Data | Backbone | Pre-train Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| MMKD |  ***no***  |  ViT-B   |  400  | 16x5x3  | [script](scripts/kd_base_ucf_slurm.sh)/[log](https://)/[checkpoint](https://) | [script](scripts/finetune_base_ucf_slurm.sh)/[log](https://)/[checkpoint](https://d) | 89.8  | 98.2  |

### Extra Data: K400

|  Method  | Datasat | Backbone | Fine-tune Epoch | \#Frame |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :---: | :---: |
| MMKD |  UCF101  |  ViT-S   |  100  | 16x5x3   | [script](scripts/finetune_small_ucf_slurm.sh)/[log](https://)/[checkpoint](https://d) | 92.7 | 99.6  |
| MMKD |  UCF101  |  ViT-B   |  100  | 16x5x3   | [script](scripts/finetune_base_ucf_slurm.sh)/[log](https://)/[checkpoint](https://d) | 96.2  | 99.6  |
| MMKD |  HMDB51  |  ViT-S   |  50  | 16x5x3   | [script](scripts/finetune_small_hmdb_slurm.sh)/[log](https://)/[checkpoint](https://d) | 72.0  | 91.1  |
| MMKD |  HMDB51  |  ViT-B   |  50  | 16x5x3   | [script](scripts/finetune_base_hmdb_slurm.sh)/[log](https://)/[checkpoint](https://d) | 77.1  | 94.1  |

### Note:

- We report the results of VideoMAE finetuned with `I3D dense sampling` on **Kinetics400**.
- \#Frame = #input_frame x #clip x #crop.
- \#input_frame means how many frames are input for model during the test phase.
- \#crop means spatial crops (e.g., 3 for left/right/center crop).
- \#clip means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).

First, create your virtual environment with 
```
virtualenv venv
```
Activate your environment
```
source venv/bin/activate
```
Download the dependencies
```
pip install -r requirements.txt
```


# Pretrain and generate Resnet-18 model with parameters using Cifar-10/100 datasets 

**step 1**
To run the training with cifar-10 datasets
```
python -m pretrain.pretrain --dataset cifar10
```

To run the training with cifar-100 datasets
```
python -m pretrain.pretrain --dataset cifar100
```

Test Accuracy is summarized as below:

| Datasets | Total Epochs (training) | Test Accuracy (last epoch) |
| --- | --- | --- |
| Cifar-10  | 150 | 93.220% |
| Cifar-100 | 200 | 77.370% |


All models with trained parameters will be saved in cifar10checkpoint folder for cifar-10 dataset
and in cifar100checkpoint folder for cifar-100 dataset
> These two folders are too large to upload to github, therefore not showing here.
> However, if the above scripts are executed, the two folders will automatically be created.
> The step below (step2) can only be conducted if the above scripts are executed. Otherwise, please skip to the step 3 where the final pretrained models is manually saved there and can be used later on.

**step 2**
Copy the last training epoch's model from each above folder to models folder (**NOTE: The following commands only works in Windows Command Prompt, for Linux System Terminal, it may be different command**)
- For cifar-10 dataset, in Windows Command Prompt, navigating to the main directory of the repo, use the following command if the last epoch's model is named as "epoch149.pt":
```
copy cifar10checkpoint\epoch149.pt models\cifar10_model.pt
```  
- For cifar-100 dataset, in Windows Command Prompt, navigating to the main directory of the repo, use the following command if the last epoch's model is named as "epoch199.pt":
```
copy cifar100checkpoint\epoch199.pt models\cifar100_model.pt
```  

**step 3**
Therefore, we now have the two final pretrained models with parameters saved in the models folder, named as below:
- cifar10_model.pt --> for cifar 10 datasets
- cifar100_model.pt --> for cifar 100 datasets


# Auto Attack to generate adversarial data

Auto Attack is performed against both cifar 10 and cifar 100 **testset**, using norm = Linf and L2 respectively for both datasets.

To generate auto-attacked adversarial data, run the following command by specifying dataset and norm:
- When dataset is cifar10 and norm = Linf, command is:
```
python auto_attack.py --dataset cifar10 --norm Linf
```
- When dataset is cifar10 and norm = L2, command is:
```
python auto_attack.py --dataset cifar10 --norm L2
```
- When dataset is cifar100 and norm = Linf, command is:
```
python auto_attack.py --dataset cifar100 --norm Linf
```
- When dataset is cifar100 and norm = L2, command is:
```
python auto_attack.py --dataset cifar100 --norm L2
```


The generated auto-attacked adversarial data is saved in auto_attack_gen_data folder:
- aa_standard_L2_cifar10_10000_eps_0.03137.pth --> uses cifar10 testset and L2 norm, with epsilon = 0.03137
- aa_standard_L2_cifar100_10000_eps_0.03137.pth --> uses cifar100 testset and L2 norm, with with epsilon = 0.03137
- aa_standard_Linf_cifar10_10000_eps_0.03137.pth --> uses cifar10 testset and Linf norm, with epsilon = 0.03137
- aa_standard_Linf_cifar100_10000_eps_0.03137.pth --> uses cifar100 testset and Linf norm, with epsilon = 0.03137


These generated auto-attacked adversarial data can be fed to previously pretrained models to test accuracy, via following command with the specification of dataset and norm:
- When dataset is cifar10 and norm = Linf, command is:
```
python auto_attacked_data_eval.py --dataset cifar10 --norm Linf
```
- When dataset is cifar10 and norm = L2, command is:
```
python auto_attacked_data_eval.py --dataset cifar10 --norm L2
```
- When dataset is cifar100 and norm = Linf, command is:
```
python auto_attacked_data_eval.py --dataset cifar100 --norm Linf
```
- When dataset is cifar100 and norm = L2, command is:
```
python auto_attacked_data_eval.py --dataset cifar100 --norm L2
```

Results are summarized as below:

| Datasets | Test Accuracy (clean data) | Test Accuracy (attacked data, Linf) | Test Accuracy (attacked data, L2) |
| --- | --- | --- | --- |
| Cifar-10  | 93.220% | 14.730% | 52.700% |
| Cifar-100 | 77.370% | 2.970% | 26.940% |

> Note: generated auto-attacked adversarial data is saved as .pth file which contains the dictionary object {'adv_complete': adv_complete}
> - adv_complete object is the adversial images generated. It is a torch.Tensor object with shape as torch.Size([10000, 3, 32, 32])
> - Note that adv_complete object only contains adversarial images' information, not the true label information.
> - Therefore, to use it together with its true label, follow steps in script auto_attacked_data_eval.py to concat it with its true labels to create a dataloader object.


# Anomaly Detection #
For Cutpaste and Reverse Distillation, the original repositories are saved at **/anomaly_detection/Cutpaste** and **/anomaly_detection/RD4AD** without any modifications. Our created scripts for training & inference are inside **/anomaly_detection**
# Cutpaste #
The plots and results are shown in **/anomaly_detection/Cutpaste_eval_plots/**

The optimal models' state dictionaries are saved in **/anomaly_detection/Cutpaste_models/** . 

Those cutpaste_cifar10_*.pth stand for training with all types(one model for one category class, only tried for cifar10)

## Training ##
```
python -m anomaly_detection.train_cutpaste [Options]
```

Options:

--epoch: number of epochs for training (default 50)

--lr: learning rate (default: 0.03)

--size: resize of input (default: 256)

--dataset: {cifar10, cifar100} (default: cifar10)

--optim: optimizer {sgd, adam} (default: sgd)

--freeze_resnet: number of epochs for freezing resnet (default: 30)

--all_type: flag for whether training one model for each category class(10 models for cifar10) rather than one model handling all (default: False)


## Inference ##
```
python -m anomaly_detection.eval_cutpaste [Options]
```
Options:

--density: using pytorch or sklearn implementation for computing Gaussian Density Score {torch, sklearn} (default: torch)

--save_plots: whether to save the T-SNE and AUROC plots (default: True)

--dataset: {ciafr10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--all_type: whether one model is for one category class(default: False) 


# Reverse Distillation #
The plots and results are shown in **/anomaly_detection/RD4AD_eval_plots/**

The optimal models' state dictionaries are saved in **/anomaly_detection/RD4AD_models/** .  Please download from this [link](https://drive.google.com/drive/folders/1dFs2KZ0FRgClwMX3i1930FAtQt4hB3OP?usp=share_link) due to the size limit on GitHub.

## Training ##

```
python -m anomaly_detection.train_rd4ad [Options]

```

Options:

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet34)

--epoch: num of epochs for training (default: 20)

--lr: learning rate (default: 0.05)

--batch_size: batch size (default: 64)

--size: resize of input (default: 64)

--dataset: {ciafr10, cifar100} (default: cifar10)

--device: {cpu, cuda} (default: cuda)


## Inference ##
```
python -m anomaly_detection.eval_rd4ad [Options]
```

Options:

--size: resize of input (default: 64)

--dataset: {ciafr10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet34)

--device: {cpu, cuda} (default: cuda)

# CutPaste Reverse Distillation (CPRD) #
The plots and results are shown in **/anomaly_detection/CPRD_eval_plots/**

The optimal models' state dictionaries are saved in **/anomaly_detection/CPRD_models/** . Please download from this [link](https://drive.google.com/drive/folders/1-aP1J_L5DX44ZPZ3isPLdAJdH6i2Ur0d?usp=share_link) due to the size limit on GitHub.


## Training ##

```
python -m anomaly_detection.train_cprd [Options]

```

Options:

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet18)

--epoch: num of epochs for training (default: 20)

--lr: learning rate (default: 0.05)

--batch_size: batch size (default: 64)

--size: resize of input (default: 64)

--dataset: {ciafr10, cifar100} (default: cifar10)

--device: {cpu, cuda} (default: cuda)


## Inference ##
```
python -m anomaly_detection.eval_cprd [Options]
```

Options:

--size: resize of input (default: 64)

--dataset: {ciafr10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet18)

--device: {cpu, cuda} (default: cuda)
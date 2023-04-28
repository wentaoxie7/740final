All codes are available https://drive.google.com/drive/folders/1zYI05BuOHdOfz434AT4q_2mAPfyOeAKJ?usp=sharing

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


All models with trained parameters will be saved in **cifar10checkpoint** folder for cifar-10 dataset
and in **cifar100checkpoint** folder for cifar-100 dataset
> These two folders are too large to upload, therefore not showing here.
> However, if the above scripts are executed, the two folders will automatically be created.
> The step below (step2) can only be conducted if the above scripts in step 1 are executed. Otherwise, please skip to the step 3 where the final pretrained models is manually saved there for further use.

**step 2**
Copy the last training epoch's model from each above folder to **models** folder (**NOTE: The following commands only works in Windows Command Prompt, for Linux System Terminal, it may be different command**)
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


The generated auto-attacked adversarial data is saved in **auto_attack_gen_data** folder:
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

--dataset: {cifar10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--all_type: whether one model is for one category class(default: False) 


# Reverse Distillation #
The plots and results are shown in **/anomaly_detection/RD4AD_eval_plots/**

The optimal models' state dictionaries are saved in **/anomaly_detection/RD4AD_models/** .  It is also available via this [link](https://drive.google.com/drive/folders/1-aP1J_L5DX44ZPZ3isPLdAJdH6i2Ur0d?usp=share_link)

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

--dataset: {cifar10, cifar100} (default: cifar10)

--device: {cpu, cuda} (default: cuda)


## Inference ##
```
python -m anomaly_detection.eval_rd4ad [Options]
```

Options:

--size: resize of input (default: 64)

--dataset: {cifar10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet34)

--device: {cpu, cuda} (default: cuda)

# CutPaste Reverse Distillation (CPRD) #
The plots and results are shown in **/anomaly_detection/CPRD_eval_plots/**

The optimal models' state dictionaries are saved in **/anomaly_detection/CPRD_models/** . It is also available via this [link](https://drive.google.com/drive/folders/1dFs2KZ0FRgClwMX3i1930FAtQt4hB3OP?usp=share_link)


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

--dataset: {cifar10, cifar100} (default: cifar10)

--device: {cpu, cuda} (default: cuda)


## Inference ##
```
python -m anomaly_detection.eval_cprd [Options]
```

Options:

--size: resize of input (default: 64)

--dataset: {cifar10, cifar100} (default: cifar10)

--dist: test on L2 or Linf norm data (default: L2)

--model: model backbone{resnet18, resnet34, resnet50, resnet152, wide_resnet50_2} (default: resnet18)

--device: {cpu, cuda} (default: cuda)



# PANDA
## Virtual Environment
Use the following commands:
```
cd PANDA-master
virtualenv venv1 --python python3
source venv1/bin/activate
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation
Use the following commands:
```
cd PANDA-master
mkdir data
```

Download:
* [80M Tiny Images - OE](https://drive.google.com/file/d/16c8-ofOnN5l7hmWp--WBCx3LIKXwHHuf/view?usp=sharing)
* [Fisher Information Matrix Diagonal](https://drive.google.com/file/d/12PTw4yNqp6bgCHj94vcowwb37m81rvpY/view?usp=sharing)

Extract these files into `PANDA-master/data` and unzip tiny.zip
## File description:
Panda.py : training PANDA
train_oe.py : training CPP
test_panda.py : testing PANDA and CPP
find_threshold.py : using anomaly scores to try to find threshold
generate_distribution.py : using anomaly scores to generate the distribution of anomaly and normal images
plot_training_auroc.pyL : using training log to plot the auroc during training
/result/[name]/data : contains the anomaly scores of test samples in every epoch
/result/[name]/model : contains the model in every epoch
/result/[name]/distances.txt : training log, each row is an epoch, each column represents:
auroc, mean of normal anomaly scores, std of normal anomaly scores, decision boundary, detect rate, accuracy, mean of anomaly score on normal images, mean of anomaly score on adversarial images
## Experiments reproduce
PANDA：
```
train:
python panda.py --dataset=cifar100  --ewc --epochs=50 --save_name=cifar100 --batch_size=32 --adversarial_address=../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth
python panda.py --dataset=cifar10  --ewc --epochs=50 --save_name=cifar10 --batch_size=32 --adversarial_address=../auto_attack_gen_data/aa_standard_Linf_cifar10_10000_eps_0.03137.pth


test:
python test_panda.py --dataset=cifar100  --ewc --epochs=49 --save_name=cifar100_L2 --batch_size=16 --adversarial_address=../auto_attack_gen_data/aa_standard_L2_cifar100_10000_eps_0.03137.pth
python test_panda.py --dataset=cifar10  --ewc --epochs=49 --save_name=cifar10_L2 --batch_size=16 --adversarial_address=../auto_attack_gen_data/aa_standard_L2_cifar10_10000_eps_0.03137.pth
```
CPP: 
```
train:
python train_oe.py --dataset=cifar100  --ewc --epochs=50 --save_name=cifar100 --batch_size=32 --adversarial_address=../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth
python train_oe.py --dataset=cifar10  --ewc --epochs=50 --save_name=cifar10 --batch_size=32 --adversarial_address=../auto_attack_gen_data/aa_standard_Linf_cifar10_10000_eps_0.03137.pth

test:
python test_oe.py --dataset=cifar100  --ewc --epochs=49 --save_name=cifar100_oe_L2 --batch_size=16 --adversarial_address=../auto_attack_gen_data/aa_standard_L2_cifar100_10000_eps_0.03137.pth
python test_oe.py --dataset=cifar10  --ewc --epochs=49 --save_name=cifar10_oe_L2 --batch_size=16 --adversarial_address=../auto_attack_gen_data/aa_standard_L2_cifar10_10000_eps_0.03137.pth

* test CPP is currently unavailable since I forgot to save the model, I will fix that problems, but it take time
```

## Experiment Result:
see in './result/cifar10/distance.txt','./result/cifar100/distance.txt','./result/cifar10_oe/distance.txt','./result/cifar100/distance.txt'





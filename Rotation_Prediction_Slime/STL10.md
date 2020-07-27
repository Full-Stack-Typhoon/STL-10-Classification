# Steps to run for STL10

## Repository Setup
```
git clone https://github.com/luke-97/fair-sslime.git
python fair-sslime/setup.py install
```

## Data preperation
```
cd "parent/folder/of/fair-sslime"
python fair-sslime/extra_scripts/create_stl10_data_files.py
```

## Training without using unsupervised data
```
# You need to edit the data paths in 'fair-sslime/extra_scripts/no_unsupervised.yaml' file
python fair-sslime/tools/train.py --config_file fair-sslime/extra_scripts/no_unsupervised.yaml
```

## Training with unsupervised data
```
# You need to edit the data and label paths in 'fair-sslime/extra_scripts/unsupervised_vgg_a_rotation_stl_10.yaml'
python fair-sslime/tools/train.py --config_file fair-sslime/extra_scripts/unsupervised_vgg_a_rotation_stl_10.yaml
# You need to edit the data, label and checkpoint model paths in 'extra_scripts/eval_vgg_a_rotation_stl_10.yaml'
python fair-sslime/tools/train.py --config_file fair-sslime/extra_scripts/eval_vgg_a_rotation_stl_10.yaml
```

## Trained models and training logs

Our trained models (unsupervised pretraining and supervised finetuning) and the training logs are available at this [link](https://drive.google.com/drive/folders/1QYv_9J8Oc2EUsQYjuCHgnxldAccKI0go?usp=sharing)

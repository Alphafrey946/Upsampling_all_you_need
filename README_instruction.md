# Instruction on how to run
The code is adopted from [Truly shift-invariant convolutional neural networks](https://github.com/achaman2/truly_shift_invariant_cnns/) by Anadi Chaman and Ivan Dokmanić. 
This code is specific for training cifar-10. 
## Training 


```python
python3 main.py --arch ARCH --filter_size j --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY --dataset_path PATH-TO-DATASET
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. Filter size ```j``` can take the values between 1 to 7. The list of CNN architectures currently supported can be found [here](https://github.com/Alphafrey946/Upsampling_all_you_need/blob/main/supported_architectures.txt). The results are saved in the path:` ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/`. 
The training weight would be saved to `ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/model`, so when creating the `CURRENT_MODEL_DIRECTORY`, make sure to also create subfolder called `model`, otherwise there would be an error. 

Vadidation would be performed after training. 

Jupyter notebook `demo.ipynb` is notebook demo for visulization of shift images and the changes in classificaiton.  

---
# Pretrained weight
Trained with CIFAR-10 with Resnet18. 
|  | Low Pass Filter| Adpative LPF | APS | Upsampling with BI | Upsampling with NN |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Weight link |[LPF](https://uwprod-my.sharepoint.com/:u:/g/personal/ydou8_wisc_edu/Ee1AACW51QtLs8MnPNXMTugBaQ-E1606xVdO31fujP7tKA?e=vhCOwc)| [Adaptive](https://uwprod-my.sharepoint.com/:u:/g/personal/ydou8_wisc_edu/EdKgEwSHHaZPvQtPNlejs8wBqjW69T5lO603YZwBQhcw9Q?e=u4S7jS) | [APS](https://uwprod-my.sharepoint.com/:u:/g/personal/ydou8_wisc_edu/ER8uLmSYCgVHvISEwRtr9i8BV8KdYC38cuawmh-_p-szTw?e=M1gxX9) | [BI](https://uwprod-my.sharepoint.com/:u:/g/personal/ydou8_wisc_edu/EZJbRwRf6f1ClYg2hxmhwFMBeR7sNrT3Xitr-zFs2Me6wA?e=GeqJjZ) | [NN](https://uwprod-my.sharepoint.com/:u:/g/personal/ydou8_wisc_edu/EfFd8bdKaSVIp7e2IpD1p4oBQIZAaAFdUS_2K9MXLMX2wA?e=0HGtar) |

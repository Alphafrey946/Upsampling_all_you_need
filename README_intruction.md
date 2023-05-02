# Instruction on how to run
The code is adopted from [Truly shift-invariant convolutional neural networks](https://github.com/achaman2/truly_shift_invariant_cnns/) by Anadi Chaman and Ivan Dokmanić. 
This code is specific for training cifar-10. 
## Training 


```python
python3 main.py --arch ARCH --filter_size j --validate_consistency --seed_num 0 --device_id 0 --model_folder CURRENT_MODEL_DIRECTORY --results_root_path ROOT_DIRECTORY --dataset_path PATH-TO-DATASET
```
```--data_augmentation_flag``` can be used to additionally train the networks with randomly shifted images. Filter size ```j``` can take the values between 1 to 7. The list of CNN architectures currently supported can be found [here](/cifar10_exps/supported_architectures.txt). The results are saved in the path:` ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/`. 
The training weight would be saved to `ROOT_DIRECTORY/CURRENT_MODEL_DIRECTORY/model`, so when creating the `CURRENT_MODEL_DIRECTORY`, make sure to also create subfolder called `model`, otherwise there would be an error. 

Vadidation would be performed after training. 

Jupyter notebook `demo.ipynb` is notebook demo for visulization of shift images and the changes in classificaiton.  

---


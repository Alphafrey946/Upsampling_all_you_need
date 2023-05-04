## Make CNN make shift invariant

CS766 Final Project

[Instruction](https://github.com/Alphafrey946/Upsampling_all_you_need/blob/main/README_instruction.md)
-------------------------
# Introduction 
Convolutional neural networks (CNNs) are a type of artificial neural network that are designed to process data with a grid-like topology, such as images or videos. They are inspired by the structure of the visual cortex in the brain and are designed to automatically learn and identify features in images.  
One of the main challenges with CNNs is their lack of shift invariance property. This property becomes critical in high stake and crucial tasks.

-------------------------

# Motivation

As previously mentioned, the shift variance exhibited by CNNs can pose a significant challenge in high-stakes classification tasks, including medical imaging analysis, object detection in autonomous driving, and surveillance and security applications. This is because even a slight shift in input images can result in drastic changes in the output of the network.  
<p align="center">
  <img src="https://user-images.githubusercontent.com/17801859/235743412-8c0e3081-d08a-4497-af28-77486be3e2bb.gif" alt="sq_shifted" width="200"/>
  <br>
  <em>[Xueyan, 2020]</em>
</p>
Above, we can observe two images of a squirrel, one of which is slightly shifted compared to the other. If we apply a CNN classifier to these images, it will produce the following results.
<div style="display:flex;flex-direction:row">
    <img src="https://user-images.githubusercontent.com/55200955/235745127-abcf3404-ec8f-4d9c-b25a-434290794992.png" width="200" />
    <img src="https://user-images.githubusercontent.com/55200955/235745156-59f04fe3-12d7-4d80-992c-59daa2090ae8.png" width="220" />
</div>  

As shown in the left bar chart, the classifier identified the non-shifted image with a high probability as a squirrel, while for the slightly shifted image (as depicted in the right bar chart), it classified it as a dog.  

In these tasks, precise object localization is critical for accurate diagnosis, navigation, and identification, and the shift invariance property of CNNs allows them to recognize objects regardless of their position or orientation in an image. Examples of such applications include the detection and segmentation of tumors in medical images, the detection and classification of objects such as cars and pedestrians in autonomous driving scenarios, and facial recognition and object tracking in surveillance and security settings.  

Therefore, in order to increase the precision and robustness of models, it is crucial to make CNNs shift invariant.

-------------------------

# Prior Works

## Data Augementation

The first proposed solution to the problem of shift variance in CNNs involved including shifted images in the training set. However, this approach still suffered from low consistency and accuracy.

## Low pass filter (LPF)

This proposed solution added a low pass filter layer before the downsampling layer, using a Gaussian filter to achieve this[1].

<p align="center">
  <img src="https://user-images.githubusercontent.com/55200955/235751982-2f6a25a9-173e-4df3-8fa1-0bcd418379d8.png" alt="sq_shifted" width="300"/>
  <br>
  <em>[Xueyan, 2020]</em>
</p>

The approach had a limitation in that it blurred certain desired high-frequency content such as edges. The issue can be observed in the above picture as well.

## Adaptive low pass filter 

This method overcomes the limitations of the previous approach by utilizing an adaptive low-pass filter [2]. The CNN predicts the weight of the Gaussian filter based on the spatial locations and channels in the image, resulting in a more precise and efficient approach.
<p align="center">
  <img src="https://user-images.githubusercontent.com/55200955/235753048-45bdf350-ca03-4229-a891-829006d4cc49.png" alt="sq_shifted" width="300"/>
  <br>
  <em>[Xueyan, 2020]</em>
</p>

In the example above, we observe that this approach allows us to maintain the desired high-frequency content. However, it also makes the network more complex, which increases computing time as we are adding more learnable parameters to the network.

## Adaptive polyphase sampling (APS)
 
The current state-of-the-art solution to the shift variance problem involves adaptive sampling grid selection to ensure that the same pixels are selected regardless of any shifts in the input images [3].

 <p align="center">
  <img src="https://user-images.githubusercontent.com/55200955/235754515-d6cbb2ef-4d1b-4573-8158-87cf2e34be44.png" alt="sq_shifted" width="450"/>
  <br>
  <em>[Chaman, 2021]</em>
</p>

By utilizing the adaptive sampling grid, the above scenario of a one-dimensional signal shows that we are able to recover the same signal even in the case of non-shifted signals.

-------------------------

# Approach

Our proposed solution builds upon the CNN with Low Pass Filter (LPF) by adding an upsampling layer after LPF and before downsampling. The purpose of the upsampling layer is to increase the output sampling rate to allow for additional content that may be introduced by subsequent layers. In our project, we experimented with two different upsampling methods: Nearest Neighbor (NN) and Bilinear Interpolation (BI).
 <p align="center">
  <img src="https://user-images.githubusercontent.com/55200955/235756743-d0b94034-309f-47b0-9b55-f61653f6f6c0.png" alt="sq_shifted" width="450"/>
  <br>
  <em>Flow of traditional LPS</em>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/55200955/235756767-0ffa5110-05b7-425e-824a-cb8476429416.png" alt="sq_shifted" width="450"/>
  <br>
  <em>Flow for our proposed solution</em>
</p>


Our proposed method can be easily integrated into existing convolutional neural networks, similar to previous works in this area.

-------------------------

# Implementation
## Upsampling + Anti-aliasing to improve shift-equivariance

Below is the implementation for BluePool by [Zhang, 2020], as indicated above.
```class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0, circular_flag = False):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.circular_flag = circular_flag    #use circular padding when this flag is on
        
        if self.circular_flag == True:
            pad_type = 'circular'

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
```
Following our proposed method abrove, we implemented our methods NN as follows:

```
    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            gauss = F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
            m = nn.UpsamplingNearest2d(scale_factor=2)
            return m(gauss)
```
And for BI
```
    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            gauss = F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
            m = nn.UpsamplingBilinear2d(scale_factor=2)
            return m(gauss)
```

Like prior works, our methods could be added to any downsampling task. For this project however, we tested our method with strided-convolution. So, we are using Resnet18 as our backbone, and training both of our method with the implementation above. [Instruction](https://github.com/Alphafrey946/Upsampling_all_you_need/blob/main/README_instruction.md) shows how to training our model. This link also includes training weight for all 5 methods. We used the same hyper-parmeters by [Chaman and Dokmanic, 2021] for training our model, or default setting in `main.py`. More specifically for our implemetnation, we use filter size of 3 for LPS and scalling factor of 2 for upsampling. 

-------------------------

# Result

## Consistency 

Consistency is a measure of the likelihood of assigning a non-shifted image and its corresponding shifted image to the same class.

## Accuracy

Accuracy is the proportion of correctly classified images out of the total number of images in the test set

## Performance comparision across all methods

To evaluate the performance of the discussed methods and our proposed method, we conducted experiments using the ResNet 18 architecture with circular padding on the CIFAR 10 dataset. The results of these experiments are summarized in the table below.

|  | Low Pass Filter| Adpative LPF | APS | Upsampling with BI | Upsampling with NN |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Accuracy | .94 | .93 | .942 | .933 | .934 |
| Consistency | .968 | .973 | 1 | .968 | .984 |

## Shift Invariance Test Results

1.) 
<div style="display:flex;flex-direction:row">
    <img src="https://user-images.githubusercontent.com/17801859/235798417-43f23e5a-f6d6-48a7-9f4c-7ebf9f9f7622.png" width="250" />
    <img src="https://user-images.githubusercontent.com/17801859/235798438-95317bfa-730c-4cdd-9098-c1753a918861.png" width="250" />
</div> 

We selected a frog image pair from the CIFAR-10 dataset to evaluate the performance of our network. The left image is the original, non-shifted image, and the right image is shifted by some pixels. 
<div style="display:flex;flex-direction:row">
    <img src="https://user-images.githubusercontent.com/17801859/235798533-8b69b397-47ab-4308-8f17-050e049d4ba6.png" width="250" />
    <img src="https://user-images.githubusercontent.com/17801859/235798573-e2007336-9577-42cd-937b-6f560f2305ec.png" width="250" />
</div> 



After feeding the images into our trained network, we generated the probability bar chart shown below for the top five classes. The bar chart reveals that our network classified both the non-shifted and shifted frog images with high probability, correctly identifying the object. However, for the shifted image, the probability of the frog class decreased slightly compared to the non-shifted image.

2.) 
<div style="display:flex;flex-direction:row">
  
   <img src="https://user-images.githubusercontent.com/17801859/235796860-d17669dd-22ed-472a-add1-6851eb35dadf.png" width="250" />

   <img src="https://user-images.githubusercontent.com/17801859/235797398-54b33399-8a17-41fe-b718-95f83ff2a08c.png" width="250" />
</div> 

The above pictures are from CIFAR 10 dataset of a ship. The left image is non shifted and right one is shifted image. We used these two images to test our network. Below we represent the bar chart for probability for first 5 classes.

<div style="display:flex;flex-direction:row">
   <img src="https://user-images.githubusercontent.com/17801859/235796854-bc8a2145-1856-4199-b8fc-6d6ef8e09c97.png" width="250" />
   <img src="https://user-images.githubusercontent.com/17801859/235796857-cfcfee47-24fb-49a7-9c39-e7141393c556.png" width="250" />
</div> 

For the non shifted image, the network classified it as ship with high probability, as evident from the bar chart. Similarly, for the shifted image, the network also classified it as boat with slightly lower probability. These results suggest that our network is able to maintain consistency in its predictions for both non-shifted and shifted images of boats.

Jupyter notebook demo:
[Notebook](https://github.com/Alphafrey946/Upsampling_all_you_need/blob/main/demo.ipynb)

-------------------------
# Dicussion

Our proposed solution, which involved upsampling with Nearest Neighbor interpolation, showed better consistency compared to two of the three prior works. However, the current state-of-the-art solution, APS, still had the best consistency among all methods.

Our proposed solution addresses certain limitations of prior works. Unlike the traditional LPF method, which tends to over-blur some of the desired high frequency content, our method does not exhibit this issue. Additionally, the adaptive LPF method introduces extra learnable parameters to the network, leading to increased training time and a more complex network. However, our proposed solution with Nearest Neighbor interpolation upsampling has better consistency as compared to these two works. Furthermore, our approach does not require any additional learnable parameters as the upsampling with interpolation is treated as an operation, resulting in faster training of the model.

In theory, we expect that our approach should be as robust as the APS method. However, the use of circular padding for generating the shifted images may introduce boundary effects. To address this issue, we use interpolation, which we hope would be able to alleviate the effect. By doing so, we aim to ensure that our approach can provide high-quality results even in the presence of shifted images. However, after running experiments our method has slightly less consistency as compared to APS.

Our results show that our proposed methods had slightly lower accuracy compared to other methods. While this finding is important, further investigation is needed to identify the specific reasons for this difference in performance. It is possible that the use of Nearest Neighbor interpolation for upsampling could lead to some loss of information, or that our approach may be more sensitive to certain types of noise or distortions. We plan to explore these factors in future work to better understand the limitations of our approach and identify opportunities for further improvement.


-------------------------
# Challenges faced

1.) One of the initial challenges we faced was finding the appropriate hyperparameters for all baseline methods and our proposed method to ensure consistency in performance comparison.

2.) Limited access to computing resources posed an additional challenge during the training of both the baseline methods and our proposed model.

3.) We encountered an unexpected challenge when our experiments revealed that Nearest Neighbor outperformed our initial assumption that Bilinear Interpolation would yield better results. This required a detailed analysis to understand the underlying reasons for the outcome, adding an extra layer of complexity to the project.

-------------------------
# Future works

1.) Provide a theoretical explanation for the slightly lower accuracy of the proposed model compared to other methods.

2.) We want to compare methods on the dataset which has more images and larger images like ImageNet and use more layered architecture like ResNet 50. The reason we want to do this is as we expect the boundary artifcats should be more prevalent in larger images.

-------------------------
# Class deliverables

1.) The project proposal could be found by clicking on the following link. [Project proposal](https://drive.google.com/file/d/1uNCe8zOJ3sSWu-LG3V0Ot3WgRjH5Tw0V/view?usp=sharing)


2.) The project mid term report could be found by clicking on the following link. [Mid term report](https://drive.google.com/file/d/1TROvKINgR0lLnnQJUksYtUtOmDQnrUff/view?usp=sharing)

3.) The project presentation could be found by clicking on the following link. [Final presentation](https://docs.google.com/presentation/d/1NkO1_GMun3ILqAEmkdj53oYfS9osN3-14Zuc3VnSx_U/edit?usp=sharing)

4.) The project presentation recording could be found by clicking on the following link. [Presentation recording](https://drive.google.com/file/d/1WBMDK910zpT2kVOaeX9stvRR4nzRQbt1/view?usp=sharing)

5.) The instructions to run the code can be found by clicking on the following link. [Instruction on running code](https://github.com/Alphafrey946/Upsampling_all_you_need/blob/main/README_intruction.md)

-------------------------
# Reference 
1.) Zhang, Richard. "Making Convolutional Networks Shift-Invariant Again." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 14205-14214.

2.) Zou, Xueyan, et al. "Delving Deeper into Anti-Aliasing in ConvNets." Proceedings of the IEEE International Conference on Computer Vision, 2022, pp 1-15.

3.) Chaman, Anadi, and Ivan Dokmanic. “Truly Shift Invariant Convolutional Neural Networks.”2021IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021

4.) Sharifzadeh, Mostafa,et al. “Investigating Shift-Variance of Convolutional Neural Networks in Ultrasound Image Segmentation”, IEEE IUS, 2021.

5.) Proakis, John G. Digital Signal Processing. Pearson, 2013.



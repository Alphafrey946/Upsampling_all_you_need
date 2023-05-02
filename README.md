## Upsampling is all you need

CS766 Final Project

-------------------------
# Introduction 
Convolutional neural networks (CNNs) are a type of artificial neural network that are designed to process data with a grid-like topology, such as images or videos. They are inspired by the structure of the visual cortex in the brain and are designed to automatically learn and identify features in images.  
One of the main challenges with CNNs is their lack of shift invariance property. This property becomes critical in high stake and crucual tasks.

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

-------------------------

# Result
![green_frog_s_001658_shift](https://user-images.githubusercontent.com/17801859/235776104-0c3aa47c-20f1-4ef9-9ede-0e5373357224.png)
![green_frog_s_001658_bar_shift](https://user-images.githubusercontent.com/17801859/235776107-03e9d23f-103a-4ff9-88ce-cd33879c02c7.png)
![green_frog_s_001658_bar_unshift](https://user-images.githubusercontent.com/17801859/235776108-149ffdc4-fdf3-43ba-8a2f-b96ec29f7256.png)
![green_frog_s_001658](https://user-images.githubusercontent.com/17801859/235776110-d1ba35e0-cef8-45ac-a8fc-933682ef0c9c.png)

![hydrofoil_s_000078_bar_shift](https://user-images.githubusercontent.com/17801859/235778167-4f270326-fe62-4b4a-999c-4c302e084951.png)
![hydrofoil_s_000078_shift](https://user-images.githubusercontent.com/17801859/235778170-e4540787-ccaf-4a54-ac61-577ae9e98316.png)
![hydrofoil_s_000078](https://user-images.githubusercontent.com/17801859/235778171-c7f40060-c764-45ea-8943-89e49914957a.png)
![hydrofoil_s_000078_bar](https://user-images.githubusercontent.com/17801859/235778173-e067940d-9105-4fea-86cc-37d06a554491.png)


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


-------------------------
# Dicussion

Our proposed solution, which involved upsampling with Nearest Neighbor interpolation, showed better consistency compared to two of the three prior works. However, the current state-of-the-art solution, APS, still had the best consistency among all methods.


-------------------------
# Challenges faced

1.) We encountered some initial challenges in training both the baseline methods and our proposed model due to limited access to computing resources.

2.) 

-------------------------
# Future works

1.) Provide a theoretical explanation for the slightly lower accuracy of the proposed model compared to other methods.

2.) We want to compare methods on the dataset which has more images and larger images like ImageNet and use more layered architecture like ResNet 50. The reason we want to do this is as we expect the boundary artifcats should be more prevalent in larger images.

-------------------------
# Reference 
1.) Zhang, Richard. "Making Convolutional Networks Shift-Invariant Again." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 14205-14214.

2.) Zou, Xueyan, et al. "Delving Deeper into Anti-Aliasing in ConvNets." Proceedings of the IEEE International Conference on Computer Vision, 2022, pp 1-15.

3.) Chaman, Anadi, and Ivan Dokmanic. “Truly Shift Invariant Convolutional Neural Networks.”2021IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021

4.) Sharifzadeh, Mostafa,et al. “Investigating Shift-Variance of Convolutional Neural Networks in Ultrasound Image Segmentation”, IEEE IUS, 2021.

5.) Proakis, John G. Digital Signal Processing. Pearson, 2013.



## Measurement of Optic Nerve Sheath on B-mode Ocular Ultrasound Image after Segmentation by Convolutional Neural Networks


#### **0. Abstract**

Recent studies have revealed the correlation between the optic nerve sheath diameter measured 3.0 mm behind the eyeball and the occurrence of elevated intracranial pressure after patient's head trauma. To conduct this measurement, in clinical diagnosis, emergency department(ED) always employ B-mode ultrasound image as a significant reference, which contains lots of pathology information not limited to ophthalmology and is non-ionizing radiation and in low-cost. However, this repetitive and onerous measurement is usually conducted by experienced physicians, and it can be finished by sophisticated computer-aided diagnosis(CAD) systems. The current state-of-the-art in medical imaging and artificial intelligence fields pays less attention to ultrasound image due to its relative indistinctness and lower resolution than other kinds of medical images like MRI, which is of high clearness. As mentioned above, the measurement needs to extract the areas of the eyeball and optic nerve sheath from the B-mode ocular ultrasound image, so we delineate a novel method of doing semantic segmentation by convolutional neural networks first, and then measuring the diameter based on the segmentation result.

***Key words***: Semantic Segmentation; Ocular Ultrasound; Deep Learning; Medical Image Analysis.

---

#### **1. Introduction**

Medical ocular ultrasonography is one of the essential techniques for clinical diagnosis \cite, and its potential applications in the emergency department(ED) includes intraocular diseases and symptoms, for example, the most common ones are retinal detachment, vitreous detachment, and vitreous hemorrhage \cite. The evaluations of them are feasible with a standard 10 MHz or 12.5 MHz linear probe \cite. Besides intraocular issues, the ocular ultrasound image often contains the optic nerve sheath part, which is believed highly related to extraocular diseases and symptoms \cite. Among them, recent studies have revealed the correlation between the optic nerve sheath diameter and the occurrence of the elevated intracranial pressure \cite. 

The elevated intracranial pressure is a common but challenging and fatal complication in neurosurgery, neurology, pediatrics and ophthalmology department \cite, usually happened after patient's acute brain trauma, intracranial tumor, cerebral hemorrhage, hydrocephalus, and intracranial inflammation \cite. As a convention, the patients require prompt and timely intervention to prevent unexpected consequences, so the physicians need to accurately distinguish the patients with the symptom of elevated intracranial pressure and those without it \cite. Direct intracranial pressure detection has invasive and non-invasive methods \cite: the invasive methods are in high accuracy and reliability but have a risk of side effects \cite. Non-invasive methods, of course, are the ideal solutions for the task, but some need to use sophisticated medical instruments and the cost is high, for example, the computed tomography(CT) scans and the magnetic resonance imaging(MRI) \cite. On the contrary, the medical ultrasonography is both almost non-invasive and in low cost \cite, which can be facilitated even in some remote region, and more importantly, the ocular ultrasound contains the optic nerve sheath, which is a good indicator of elevated intracranial pressure.

In this paper, we will present a novel method of exploiting the B-mode ocular ultrasound image for measurement of optic nerve sheath diameter(ONSD), based on the semantic segmentation by convolutional neural networks. Since there are few studies on ocular ultrasound with deep learning and we can hardly find public datasets on the Internet, we obtained the dataset from the *Beijing Tongren Hospital*, which consists of *six video*s of ocular ultrasound. We sample these six videos by frames, and the frames that do not contain a precise shape of the eyeball are eliminated. Then we manually annotate each image with the areas of the eyeball and optic nerve sheath and train a convolutional neural network model which can segment the areas of the eyeball and optic nerve sheath. Based on the segmented areas, we can conduct the measurement of ONSD at 3.0 mm behind the eyeball.

<p align = "center"><img width = "40%" height = "40%" src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/00.png" /> </p>

We do not intend to study the correlation between ONSD and elevated intracranial pressure, but to propose a deep learning based method that may assist clinical physicians on the measurement.

---

#### **2. Data Preparation**

After sampling and eliminating invalid images from the video, we get 419 images to form the original training dataset. Each image is gray-scale with the resolution of 384×384. The following preprocessing steps are the preparation for training the neural network model.

**2.1 Manual Annotation**

The images from the dataset are not labeled while obtaining from the hospital, and to apply supervised deep learning method we need the annotation information with areas of the eyeball and optic nerve sheath. Since the size of the dataset is small and the annotation work for each image is attainbable, after removing artifacts, we manually annotate each image with areas of the eyeball and optic nerve sheath.

<p align = "center"><img width = "60%" height = "60%" src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/02.png" /> </p>

The annotation includes two channels: red channel denotes the area of eyeball, and the green channel denotes the area of the optic nerve sheath. As for a semantic segmentation task, the segmentation can be regarded as three objects: eyeball, optic nerve sheath, and the background. However, it can also be regarded as only two concerned objects, eyeball, and optic nerve, and it depends on the structure of the neural network model.

**2.2 Data Augmentation**

To make the deep learning model generalize better, it had better train it on more available data \cite. Empirically, the original size of the dataset is insufficient \cite. To figure out this problem, we adopt the method of dataset augmentation which aims to generate new different data from the original dataset, since image data are of high dimension and contain massive variable factors, which means the attribute of being stimulated easily \cite.

The simplest and the most commonly used method of dataset augmentation is geometric transformation \cite. Specifically speaking, it includes the following operations:

* Reflection
* Rotation
* Translation
* Scaling

In the augmentation process, we apply both single and multiple operations to each image and generate 24 different ones from each image. As a result, the size of the dataset is enlarged into 419×24=10056, approximately ten thousand images. This quantity scale we think is applicative to this task \cite.

In addition to the method mentioned above, a more sophisticated approach is the recently proposed generative adversarial networks(GANs), which is based on the unsupervised learning algorithm to generate fake data similar to the real training data \cite. GAN can also play an essential role in dataset augmentation process in object recognition or analogous tasks within which the labels are one dimensional and easy to annotate manually \cite. As semantic segmentation task, the image-to-image translation conditional GAN model has also been introduced to generate the image data with segmentation label simultaneously \cite.

**2.3 Neural Network**

The method of deep learning has been now applied in almost every image related field \cite since the Alexnet was proved to break the state-of-the-art object recognition task \cite. The main contributor of deep learning in image processing field is the convolutional layer \cite since it can effectively extract the features from images not limited to specific positions. The deep learning method is to train a very large neural network model with millions or even billions of parameters, and different fields have different compatible models for specific tasks \cite. As for medical image analysis field, research is always advanced compared to other fields, because the demand for medical development and healthcare is very strong and urgent, and the research in this field has irreplaceable practical values \cite. Typical uses of deep learning methods in medical image analysis include classification, object detection, segmentation, registration and so on, each of which can achieve great results with suitable neural network models and large training dataset \cite.

In medical image segmentation field, we concern in this paper, the U-Net model is the recent state-of-the-art \cite, and there are also derivative neural networks based on U-Net, like U-Net++ \cite and V-Net \cite, each of which have achieved very high accuracy in different medical image segmentation tasks. We may be the first to apply the U-Net model on ocular ultrasound images. The structure of U-Net consists of encoder layers, decoder layers, and the concatenation layers to merge each encoder layer with its corresponding decoder layer. One of the most important structural parameters in U-Net is the depth, which indicates the complexity and the ability to extract features. The depth of U-Net is always equal to the number of encoder or decoder layers, and exponentially correlated to the number of trainable parameters. In other words, the greater the depth of U-Net, the better the performance and the higher the accuracy but, the longer the necessary training time.

Under general considerations, we use the U-Net with the depth of four, and it is also the most widely-used one \cite.

---

#### **3. Configuration**

The images and corresponding segmentation labels from the augmented dataset are served as input and ground truth to train the U-Net model. For efficiency, we also resize the images and the segmentation labels from 384×384 to 96×96, and the following experiment results show that this operation does not affect the final performances. We realize the U-Net model with currently the most popular and user-friendly deep learning framework, Keras (with Tensorflow at backend) \cite in Python, and pipeline of the entire experiment is also entirely in Python environment. 

**3.1 Dataset Split**

From the approximately ten thousand images in the dataset, we divide them into three subsets randomly: training set, validation set, and test set, and their sizes are 8000, 1000, 1000, respectively. The training set is to train the parameters in the U-Net to reduce the loss value and make the prediction results close to the segmentation ground truths. Then, the validation set is to monitor the performance metrics while training, and if necessary we should manually adjust the hyper-parameters, like learning rate, to achieve better final performances as well as avoid the overfitting problem \cite. At last, the testing set is to evaluate the ability of the model after the training finish.

**3.2 Standardization**

The training of deep neural network model is always intricate since the distribution in each layer changes as the parameters of the previous layers change \cite. To accelerate the training process, data normalization, which makes the input data obey the assumption of being $i.i.d.$, can be applied before input. In the experiments, we adopt a common normalization method, zero-mean  and unit-variance normalization \cite, whose formula shows below:
$$
\begin{equation}
\begin{aligned}
\mu&=\frac{1}{MN}\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}X_{i,j} \\
\sigma&=\sqrt{\frac{1}{MN}\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}(X_{i,j}-\mu)^2}\\
\mathbf{X'}&=\frac{\mathbf{X}-\mu}{\sigma}
\end{aligned}
\end{equation}
$$
Where $\mu$ is the arithmetic mean of input image $\mathbf{X}$, and the $\sigma$ is the standard deviation of input image $\mathbf{X}$. After applying this, all the input images are $i.i.d.$ and obey the distribution $N(0,1)$.

**3.3 Dropout**

The original structure of U-Net does not contain the dropout layer \cite. Adding dropout layers in the model have been testified a very useful technique of preventing the overfitting problem \cite. To find the best model, we compose two versions of U-Net, one of which is with two dropout layers at the bottom following the convolutional layers, and the other one is the same as original U-Net without dropout layers.

<p align = "center"><img src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/03.png" /> </p>

The dropout layers are optional, and both of these two kinds of models will be trained under the same condition, to compare the generalization. Note that the last two layers, including the output layer, can be either 96×96×2 or 96×96×3, which depends on which loss function is applied. If the two-channel version is adopted, the segmentation label ground truth will also be two-channel, including the red and green channel, and the blue channel will be ignored. On the other hand, if we use the three-channel version, the ground truth will be three channel, and besides the red and green channel, the third channel that denotes the background area will be the complementary set of the union of the red and green area, or to say, the black area.

**3.4 Loss Function**

After configuring the structure of U-Net, the next step is to determine to use which loss function. The loss function is always the key to the success of neural network training \cite since the training process is to reduce the loss value between the prediction and the ground truth. We suppose three kinds of loss functions to compare in the following experiments. No matter what kind of loss function adopted, the optimizer we choose is the `Adam`, which is also the current state-of-the-art \cite.

3.4.1 Mean Squared Error (MSE)

MSE is frequently used in regression machine learning tasks \cite. Under this condition, the U-Net will be two-channel version: the first output channel denotes the possibility of being the area of eyeball, while the second output channel denotes the possibility of being the area of the optic nerve sheath, and the two channels are estimated independently.
$$
\begin{equation}
\begin{aligned}
L_{mse}&=\frac{1}{MN}\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}(\widehat{Y}_{i,j}-Y_{i,j})^2\\
\end{aligned}
\end{equation}
$$
Since the output has two channels, the loss of the entire model will be calculated by the average value of the MSE between these two channels.

3.4.2 Dice Coefficient

Dice Coefficient is a statistic used to estimate the similarity of two samples \cite, whose original formula shows as below:
$$
Dice=\frac{2|\mathbf{X}\cap\mathbf{Y}|}{|\mathbf{X}|+|\mathbf{Y}|}
$$
Also, it can be used to calculate the similarity between the two areas. Thus, in semantic segmentation tasks, it is a potent technique to serve as the loss function to optimize the parameters from the model \cite. In practice, the Dice Coefficient and its corresponding loss function will be:
$$
\begin{aligned}
Dice&=\frac{2\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}\widehat{Y}_{i,j}\cdot Y_{i,j}+s}{\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}(\widehat{Y}_{i,j}^{2}+Y_{i,j}^{2})+s} \\
L_{dice}&=1-Dice
\end{aligned}
$$
The same as MSE, when using dice coefficient loss as loss function, the U-Net will also be two-channel version: the two channels of output are independent of each other, and each denotes the possibility of its corresponding object. The loss of the whole model is also the average value of the Dice Coefficient Loss between two output channels.

3.4.3 Categorical Cross Entropy

In general, the loss function of cross entropy is used in classification tasks \cite (e.g., ImageNet contest \cite). If the number of classification objects is two, we usually call it binary cross entropy \cite, and if the number is greater than two, we call it categorical cross entropy. The semantic segmentation task can also be regarded as a classification task, which is a pixel-wise one. In our experiments, the classification will be conducted pixel-by-pixel, into three main categories: eyeball, optic nerve sheath, and background. Therefore, the loss function of categorical cross entropy can be applied in this task, and under this condition, we need to use the three-channel version of U-Net: each of the three output channels denotes the possibility of a corresponding classification object. The formula will be:
$$
\begin{aligned}
L_{cce}&=-\frac{1}{MN}\sum\limits_{i=1}^{M}\sum\limits_{j=1}^{N}(Y_{i,j}\cdot \mathrm{log}\widehat{Y}_{i,j})
\end{aligned}
$$
The loss of the whole model is also the average value of categorical cross entropy among three output channels. Note that these three channels are not independent of one another since the activation of the last layer will be replaced with `softmax`, and the sum of the possibility of the three channels at every pixel will be identically equal to one.

**3.5 Early Stop**

In the pre-experiments, we find the overfitting problem is inevitable when the number of training epochs is great enough, roughly 300, so we need to do the early stop to obtain the best model while training \cite. The Keras framework has a beneficial component called *callback*, which is a set of functions to monitor the internal states and statistics of the model while training \cite. Among them, we deploy the `keras.callbacks.ModelCheckpoint` function to the initial training configuration and it saves the latest model to `.h5` file while training if the appointed value, such as the loss value on the validation set, is less then ever before. In our experiments, only the model that had the lowest loss value on the validation set will be saved, and the training process after the best model will be overlooked.

**3.6 Evaluation**

The evaluation of each model will focus on the accuracy of the two output channels. Even though in the categorical cross entropy condition, the output has three channels, only the channels of the eyeball and optic nerve sheath will be put into compared, since the three channels are not independent of one another and we only concern the performances on the two concerned objects: eyeball and optic nerve sheath. The particular calculation will be the intersection over union (IoU) method \cite.
$$
IoU=\frac{|\mathbf{Y}\cap\widehat{\mathbf{Y}}|}{|\mathbf{Y}\cup\widehat{\mathbf{Y}}|}
$$


The evaluation will be conducted on the test dataset, which is an important reference for estimating the generalization ability of a model \cite.

---

#### 4. U-Net Training

We have purported two kinds of U-Net, with and without Dropout layers, and three kinds of loss possible loss functions, so, altogether we have six different experiments situations. We will run these experiments on a cloud computing cluster provided by the `LEINAO.ai`, along with a single **NVIDIA Tesla V100** Tensor Core GPU. In the beginning, the parameters in the model will be initialized by the `he_normal` method \cite, and the training process each time has 500 epochs, but the learning rate will be manually adjusted to fit in to ensure the only the best models from each condition will be compared.

The training curves of  each condition show as below:

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| (1) MSE w/o D                                                | (2) MSE w D                                                  |
|                                                              |                                                              |
| (3) Dice w/o D                                               | (4) Dice w D                                                 |
| ![cce](https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/curve_cce_d.png) | ![cce_d](https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/curve_cce.png) |
| (5) CCE w/o D                                                | (6) CCE w D                                                  |

We can find that in all of the six conditions the overfitting problem always occurs, but the dropout layers have successfully deferred the occurrence of overfitting. Only the best models will be saved when the loss value on the validation set is smallest, to conduct the comparison of the performances on the test dataset.

| IoU        | Eyeball (%) | Optic Nerve Sheath(%) | Average(%) |
| ---------- | ----------- | --------------------- | ---------- |
| MSE w/o D  |             |                       |            |
| MSE w D    |             |                       |            |
| Dice w/o D |             |                       |            |
| Dice w D   |             |                       |            |
| CCE w/o D  |             |                       |            |
| CCE w D    | 97.76       | 93.06                 | 95.41      |

We find that the U-Net model using the loss function of categorical cross entropy with dropout layers has achieved best in IoU accuracy on the test set.

<p align = "center"><img width = "80%" height = "80%" src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/05.png" /> </p>

The segmentation predictions are very close to the ground truths: even the naked eye cannot tell the differences between them. This results will be the foundation of the measurement of ONSD.

---

#### 5. ONDS Measurement

After completing the segmentation process, we can conduct the measurement of ONDS based on the segmentation prediction of each ocular ultrasound image. 

**5.1 Pixel Resolution**

The first step is to determine the pixel resolution, which stands for the actual size of each pixel in the images. Fortunately, the raw data from the hospital includes the scale, so that we can easily calculate the pixel resolution.

<p align = "center"><img width = "60%" height = "60%" src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/06.png" /> </p>

Every 128 pixels represent 1 cm, or 10 mm.  During the data preparation, besides the resize process which scales the image from 384×384 to 96×96, the pixel scale of the image remains unchanged. So the final pixel scale will be $\frac{1}{128}cm/px\cdot4=\frac{5}{16}mm/px$. in other words, the length of each pixel is $\frac{5}{16}mm$.

**5.2 Location**

According to the references \cite, the ONDS should be measured 3.0 mm behind the eyeball. We need to locate the exact position of this point. $\frac{3.0mm}{\frac{5}{16}mm/px}=9.6px$, which means that the point is 9.6 pixels behind the eyeball in the image. Then, we connect the geometric centers of areas of the eyeball and optic nerve sheath to form an axis. The point where the axis intersects the contour of the eyeball is regarded as original point, and the point 9.6 pixels from the original point on the optic nerve sheath side will be the exact position where the ONSD is measured.

**5.3 Measurement**

The ONSD will be the segment normal to the axis, and whose two endpoints are the intersection on the contour of the area of the optic nerve sheath.

<p align = "center"><img src = "https://raw.githubusercontent.com/taotie144/ImgStore/master/temp6/07.png" /> </p>

The whole pipeline of the measurement: (a) Ocular ultrasound image to input; (b) The trained U-Net model. (c) The prediction of segmentation. (d) Contours extracted. (Here we turn the background to white to show more clearly.) (e) Find the geometric centers of areas of the eyeball and optic nerve sheath, point $A$ and $B$, respectively. (f) Segment $\overline{AB}$ intersects the contour of the eyeball at point $C$, and the length of $\overline{CD}$ is 3.0 mm, and point $D$ is where the measurement should be conducted. (g) The segment $\overline{EF}$ is normal to $\overline{AB}$ and intersects the contour of optic nerve sheath at point $E$ and $F$. The ONSD is equal to the length of $\overline{EF}$.

In this sample, ONSD, the length of $\overline{EF}$ is 4.38mm. Hereto, we have succeeded in the measurement of ONSD.

---

#### 6. Conclusion

The actual relationship of the ONSD and the occurrence of elevated intracranial pressure is far more intricate. Though some papers claim that 5.0 mm may be the threshold, it is just an assumption, and we cannot guarantee the existence of this threshold. In clinical diagnosis, other techniques are necessary for more precise results. To figure out this correlation, more clinical data and more sophisticated analysis methods are indispensable. 

This only purpose of this paper is to introduce a novel method involving with deep learning of the measurement of ONSD. We can assert that the U-Net model is perfect for the semantic segmentation task, which has achieved the IoU accuracy of about 95.41%. The manual measurement of ONSD is hard to induce practical principles, so we also introduce a method of measurement based on the segmentation results, and this can also be considered successful.

---

#### **7. Acknowledgement**

This study was assisted by the *Beijing Tongren Hospital*, who provided us the medical instrument and dataset, and `LEINAO.ai`, who provided us the cloud computing platform with GPU.

---

#### 8. References

TODO.
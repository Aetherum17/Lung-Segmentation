# Lung Segmentation
 
* Lung segmentation is the process of accurately identifying regions and boundaries of the lung field from surrounding thoracic tissue.
* The COVID-19 radiography dataset was used for model training and testing. This dataset is a collection of chest X-ray images of patients with COVID-19, along with images of patients with other lung diseases such as pneumonia. The
dataset was created by researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh. It is freely available on the Kaggle website.
* To perform the segmentation task, the U-Net Deep Learning model was selected. Its performance was estimated using the Average Dice Coefficient and Average Intersection over Union metrics.
* The U-Net model was trained for 1/3/5 epochs with different learning rates of the Adam optimizer until its accuracy reaches an accuracy close to 1.0 or until there is no improvement in model accuracy/loss decrease was observed. The results of the U-Net model are
summarised in Table 1. The examples of predicted masks with respective original images and grand truth masks for the U-Net model with Adam optimizer with the default learning rate are presented in Figure 1.
* It was discovered that different learning rates play an important role in
image validation accuracy by the U-net model. At first glance, we can observe that the accuracy of the model is
fairly high in general. However, the model reaches the best performance when the learning
rate is below 0.001. It might be possible that the reason for this is that the model does not have a too complex
structure, so unlike other deep learning architectures, it converges very quickly. Therefore, a small learning rate is required for the model to find the best weights.

![image](https://github.com/Aetherum17/Lung-Segmentation/assets/46795020/432e342c-40b5-45a8-b608-4cea3c1ebac2)

![image](https://github.com/Aetherum17/Lung-Segmentation/assets/46795020/f12cef47-f15b-48d7-859e-50061dd32486)

Figure 1. The U-Net model predicted masks (Adam optimizer) with respective original images and grand truth masks.

<h1 align="center">Face Mask Detection</h1>

<div align= "center">
  <h4>Face Mask Detection system built with OpenCV, Keras, TensorFlow using Convolutional Neural Network approach to detect face masks including Face ROI in static images as well as in real-time video streams.</h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/nadhirfr/face_mask_detect/issues)





## :point_down: Support me here!
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/H2H146AUD)


## :warning: Framework used

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

## :star: Features
This face mask detection experiment use 3 different datasets with 2 different models architecture. More details in [Dataset](#dataset) and [Models](#models) section respectively. The most accurate model, which has **100% accuracy**, is trained using **dataset_1** with the **MobileNetV2** architecture.  Using MobileNetV2 architecture also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, NVidia Jetson, etc.). More comparative detail in Result section.

This system can be used in real-time applications which require face-mask detection. This project can also be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

## :file_folder: Dataset
There are three different dataset used. All the dataset could can be downloaded directly in this repository on each folder respectively. The images were collected from the following sources:

* __dataset_1__ ([See The Source](https://github.com/prajnasb/observations/tree/master/experiements/data))
In this first dataset, an image of a person using a mask is deliberately created, or it is called an augmented image. The image of the mask is placed between the nose and the chin matches the position of the person in the image. The dataset consist of 1376 pictures beloging into two classes:
__with_mask__: 690 pictures
__without_mask__: 686 pictures

* __dataset_2__ ([See The Source](https://www.kaggle.com/andrewmvd/face-mask-detection))
The dataset used original images of people using different types of masks. This dataset consists of a total of __853 images__. Unlike the previous dataset, in this dataset one image can be included in both classes because in one image there can be many images of human faces. The labels on each face in the image are stored in an annotation in the __xml__ file.  From those xml file information, cropping can be done to save faces with or without masks. From the cropping results, the number of classes __with_mask is 3355 images__ and __without_mask is 717 images__.

* __dataset_3__ ([See The Source](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset))
This dataset contains about 1006 equally distributed images of 2 distinct types:
__with_mask__: 503 pictures
__without_mask__: 503 pictures

## :bulb: Models
The experiment of this system are using two architecture model:

 1. CNN
    The first model uses a general convolutional layer. Input images in 100x100 size with a grayscale transform. Followed by two Conv2D layer with 200 and 100 filter size respectively. Then using fully connected layer (Dense 64) and (Dense 2). Below is the view of the neural network layer:

  ![layer_cnn](https://github.com/nadhirfr/face_mask_detect/blob/main/result/layer_cnn.svg)


 3. MobileNetV2 CNN

    The second model uses MobileNetV2 as base model. Input image in 224x224 RGB. The output from MobileNetV2 comes into AveragePooling then connected to fully connected layer (Dense 128) and (Dense 2) for a probability beetween with_mask or without_mask.

    ![layer_cnn_mobilenetv2](https://github.com/nadhirfr/face_mask_detect/blob/main/result/layer_cnn_mobilenetv2.svg)

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/nadhirfr/face_mask_detect/blob/master/requirements.txt)

## 🚀&nbsp; Installation
1. Clone the repo
```
git clone https://github.com/nadhirfr/face_mask_detect
```
2. Change your directory to the cloned repo 
```
cd face_mask_detect
```

3. Now, run the following command in your Terminal/Command Prompt to install the libraries required

```
pip install -r requirements.txt
```

## :bulb: Run the Detection

1. To detect face masks in an image type the following command: 

```
python detect_mask_image.py --image "path/to/image.jpeg" --model "trained_model_1.h5"
```

To detect mask in an image with MobileNetV2 models use following command:

```
python detect_mask_image_MobileNetV2.py --image "path/to/image.jpeg" --model "trained_model_1_MobileNetV2.h5"
```

2. To detect face masks in real-time video streams type the following command:

```
python detect_mask_video.py --model "trained_model_1.h5"
```
To detect face masks in real-time video streams with MobileNetV2 models type the following command:

```
python detect_mask_video_MobileNetV2.py --model "trained_model_1_MobileNetV2.h5"
```



Use script and trained models **ended with MobileNetV2** respectively to **run MobileNetV2** models. The number in trained models name indicates the dataset used, for example:
**"trained_model_1_MobileNetV2.h5"**

this trained model is trained using **dataset_1** with **MobileNetV2**



## :key: Results

#### We have done with total 6 combination of training, below is the detail:

| Model           | Dataset   | Training Notebook                                            | Trained Model                                                | Accuracy |
| --------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| CNN             | dataset_1 | [train_mask_detector_1.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_1.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_1.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_1.h5) | 88%      |
| CNN             | dataset_2 | [train_mask_detector_2.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_2.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_2.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_2.h5) | 92%      |
| CNN             | dataset_3 | [train_mask_detector_3.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_3.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_3.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_3.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_3.h5) | 85%      |
| MobileNetV2 CNN | dataset_1 | [train_mask_detector_1_MobileNetV2.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_1_MobileNetV2.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_1_MobileNetV2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_1_MobileNetV2.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_1_MobileNetV2.h5) | 100%     |
| MobileNetV2 CNN | dataset_2 | [train_mask_detector_2_MobileNetV2.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_2_MobileNetV2.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_2_MobileNetV2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_2_MobileNetV2.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_2_MobileNetV2.h5) | 90%      |
| MobileNetV2 CNN | dataset_3 | [train_mask_detector_3_MobileNetV2.ipynb](https://github.com/nadhirfr/face_mask_detect/blob/main/train_mask_detector_3_MobileNetV2.ipynb)  <a href="https://colab.research.google.com/github/nadhirfr/face_mask_detect/blob/main/train_mask_detector_3_MobileNetV2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [trained_model_3_MobileNetV2.h5](https://github.com/nadhirfr/face_mask_detect/blob/main/trained_model_3_MobileNetV2.h5) | 99%      |

The best model result is used the dataset_1 with the MobileNetV2 CNN which has 100% accuracy.  Below is the graph and classification report:

![best_result_accuracy_graph](https://github.com/nadhirfr/face_mask_detect/blob/main/result/best_result_accuracy_graph.png)

![best_result_classification_report](https://github.com/nadhirfr/face_mask_detect/blob/main/result/best_result_classification_report.png)

To see the other accuracy graph and classification report kindly open the training notebook in Google Colab.



## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: nadhir.rozam@gmail.com



## :+1: Credits
* https://ieeexplore.ieee.org/document/9342585
* https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
* https://www.tensorflow.org/tutorials/images/transfer_learning
* https://github.com/chandrikadeb7/Face-Mask-Detection
* https://github.com/prajnasb/observations/tree/master/experiements/data
* https://www.kaggle.com/andrewmvd/face-mask-detection
* https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset




## :raising_hand: Citation

You are allowed to cite any part of the code or our dataset. You can use it in your Research Work or Project. Remember to provide credit to the Maintainer Nadhir Rozam by mentioning a link to this repository and her GitHub Profile.

Follow this format:
- Author's name - Nadhir Rozam
- Date of publication or update in parentheses.
- Title or description of document.
- URL.


## :eyes: License
MIT © [LICENSE](https://github.com/nadhirfr/face_mask_detect/blob/main/LICENSE)
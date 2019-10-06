[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses



## Project Instructions

All of the starting code and resources you'll need to compete this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project. If you have already created a `cv-nd` environment for [exercise code](https://github.com/udacity/CVND_Exercises), then you can use that environment! If not, instructions for creation and activation are below.

*Note that this project does not require the use of GPU, so this repo does not include instructions for GPU setup.*


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the P1_Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.


## Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd P1_Facial_Keypoints
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-nd` environment by clicking `Kernel > Change Kernel > cv-nd`.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality and answer all of the questions included in the notebook. __Unless requested, it's suggested that you do not modify code that has already been included.__


## Evaluation

Your project will be reviewed against the project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

When you are ready to submit your project, collect all of your project files -- all executed notebooks, and python files -- and compress them into a single zip archive for upload.

Alternatively, your submission could consist of only the **GitHub link** to your repository with all of the completed files.

<a id='rubric'></a>
## Project Rubric

### `models.py`

#### Specify the CNN architecture
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a CNN in `models.py`. |  Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). The network should take in a grayscale, square image. |


### Notebook 2

#### Define the data transform for training and test data
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a `data_transform` and apply it whenever you instantiate a DataLoader. |  The composed transform should include: rescaling/cropping, normalization, and turning input images into torch Tensors. The transform should turn any input image into a normalized, square, grayscale image and then a Tensor for your model to take it as input. |

#### Define the loss and optimization functions
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Select a loss function and optimizer for training the model. |  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem. |


#### Train the CNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Train your model.  |  Train your CNN after defining its loss and optimization functions. You are encouraged, but not required, to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Save your best trained model. |


#### Answer questions about model architecture

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| All questions about model, training, and loss choices are answered.  | After training, all 3 questions in notebook 2 about model architecture, choice of loss function, and choice of batch_size and epoch parameters are answered. |


#### Visualize one or more learned feature maps

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Apply a learned convolutional kernel to an image and see its effects. |  Your CNN "learns" (updates the weights in its convolutional layers) to recognize features and this step requires that you extract at least one convolutional filter from the trained model, apply it to an image, and see what effect this filter has on the image. |


#### Answer question about feature visualization
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  After visualizing a feature map, answer: what do you think it detects? | This answer should be informed by how the filtered image (from the step above) looks. |



### Notebook 3

#### Detect faces in a given image
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Use a haar cascade face detector to detect faces in a given image. | The submission successfully employs OpenCV's face detection to detect all faces in a selected image. |

#### Transform each detected face into an input Tensor
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Turn each detected image of a face into an appropriate input Tensor. | You should transform any face into a normalized, square, grayscale image and then a Tensor for your model to take in as input (similar to what the `data_transform` did in Notebook 2). |

#### Predict and display the keypoints
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Predict and display the keypoints on each detected face. | After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, and display the predicted keypoints on each face in the image. |

LICENSE: This project is licensed under the terms of the MIT license.

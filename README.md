# csci4270-computational-vision
Welcome to my CSCI4270 Computational Vision course repository! This collection contains six detailed Jupyter Notebooks or PDFs that walk through key concepts in image processing and computational perception, implemented using Python, NumPy, and OpenCV.

---

## hw1: Image Manipulation

**Location:** `hw1/hw1_start.ipynb`

### Description
Using cv2, numpy, and matplotlib to read in images, transform and color maipulation, and then displaying the alterred image:

- Image rotation and duplication
- Applying a vignette that decreases RGB values the further away the pixel is from a given point
- Downsizing images by altering them to black and white block images

## hw2: Homogeneous Camera Matrix 

**Location:** `hw2/hw2_start.ipynb`

### Description
Camera representation and matrix manipulation: 

- Creating a representation of a camera in 3d space via the homogeneous camera matrix and identifying if a given point in the 3D space is visible to the camera
- Implementing the RANSAC algorithm for fitting a line to a set of points
- Identifying the best focused image in a given image directory by applying Sobel kernel then calculating average squared gradient magnitude across all images

## hw3: Multi-Image Mosaics

**Location:** `hw3/hw3_align.ipynb`

### Description
Creating mosaics of two images from similar scenes in a given image directory:

- SIFT keypoint detection to identify if two images are from the same scene
- Identifying the fundamental matrix using RANSAC
- Computing the homography matrix to identify translation from one image to its similar scene counterpart
- Warping image to overlap then applying a blend

## hw4: Classification and Deep Learning

**Location:** 

`hw4_part1_start.ipynb`

`hw4_part1_start.ipynb`

### Description
Part 1: k-NN classification of Fashion MNIST

- Visualize class distributions and tune hyperparameters
- Implement accuracy metrics and confusion matrix

Part 2: Scene classification with a CNN using PyTorch
- Train on grass, road, ocean, etc.
- Track validation performance and test accuracy
- Visualize learning curves and confusion matrix

## hw5: Object Detection with Region Proposals

**Location:** `hw5_start.ipynb`

### Description
Taking an image directory to detect objects as well as apply bounding boxes for the identified objects:

- Use pretrained ResNet-18 as a frozen backbone
- Add classification and bounding box regression heads
- Train on a dataset with multiple candidate regions
- Implement combined loss function (cross-entropy + regression)
- Post-process detections with non-maximum suppression
- Evaluate using mean Average Precision (mAP)
- Visualize correct/incorrect/missed detections with color-coded boxes

## hw6: Motion Estimation and Moving Object Detection

**Location:** `hw6.ipynb`

### Description
Finding the movement of a camera based on a given image pair:

- Estimate sparse optical flow using Harris corners and KLT tracking
- Determine camera motion via focus of expansion (FOE)
- Identify independently moving objects in scenes with a moving or static camera
- Cluster detected motion outliers into coherent object groups
- Visualize FOE and object tracks, bounding detected objects


# ML_Project_GoogleMaps_ImagesAI

## Deep Learning Project: Street View Housing Number Image Recognition of Digit

The project was assigned by mentors at MIT Professional Education: Applied Data Science program in 2023. I rephrased and added in my insights and findings in so it's more applicable to the keywords search, and reveal my analysis and understanding of the Deep Learning process.

**Context**
One of the most interesting tasks in deep learning is to recognize objects in natural scenes. The ability to process visual information using machine learning algorithms can be very useful as demonstrated in various applications.

The SVHN dataset contains over 600,000 labeled digits cropped from street-level photos. It is one of the most popular image recognition datasets. It has been used in neural networks created by Google to improve the map quality by automatically transcribing the address numbers from a patch of pixels. The transcribed number with a known street address helps pinpoint the location of the building it represents.

**Objective**
Our objective is to predict the number depicted inside the image by using Artificial or Fully Connected Feed Forward Neural Networks and Convolutional Neural Networks. We will go through various models of each and finally select the one that is giving us the best performance.

**Dataset**
Here, we will use a subset of the original data to save some computation time. The dataset is provided as a .h5 file. The basic preprocessing steps have been applied on the dataset.

**Machine Learning - Deep Learning Process**
The neccessay libraries and Python packages I use for this project are: numpy, pandas, seaborn, matplotlib, sklearn, tensorflow.keras

Steps that I took:
1. Connect to cloud drive, working environment
2. Import Python packages and libraries
3. Load the data set for multidimensional array of data elements.
4. Split the data into train and test datasets.
5. Visualized data images for labeled data.
6. Prepare, clean, shape the data so it fits the input of the neural network model.
7. Convert a class vector into binary class
8. Build the First Model Architecture
9. Build and train the Artificual Neural Network 1
10. Plot the validation and training accuracies of ANN 1 for training dataset
11. Build the Second Model Architecture 
12. Build and train the Artificual Neural Network 2
13. Plot the validation and training accuracies of ANN 2 for test dataset
14. Predict on test dataset
15. Build the classification report and the confusion matrix
16. Intepret the results

Steps 17+: I would include the above steps with Convolutional Neural Network Model 1 and 2 for the same dataset.

**Results**
I compare all the test results, and select the best predictive model based on the results of accuracy matrix and my scientific judgement. Build a report with intepretative results, insights and recommendations.

Please see other files in this folder for further information and results.

# Feature Sieve Regression Experiments
## Dataset: UTKFace

### Dataset Infos:
Dataset consists of total of 23705 datapoints.\
Divided into:
1) Training Dataset of 14223 datapoints.
2) Validation Dataset of 4741 datapoints.
3) Test Datset of 4741 datapoints

To experiment on simplicity bias, we create biased datasets by biasing the dataset using the attribute gender.\
We have 7470 male datapoints and 6753 female datapoints.

Here is the image of the training dataset distribution, where blue represents the gender male and red represents the gender female.\
![Training Distribution as a Violin Plot](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/train_distribution.png)

#### Data Generation 1
Here we are biasing the dataset where we have the target as *Age* and input as *Image* and biasing attribute as *Gender*.\
The function takes in the following parameters for creating the biased dataset:
1) **Threshold** - Age value to separate the data points of the two genders
2) **Percentage** - Percentage of male data points in the total dataset, the rest is filled with female data points.
3) **Old** - To indicate which gender takes the older part of the spectrum.

Here are the values used to create the 10 datasets\
*Threshold* : 25, 20, 23, 23, 25, 27, 39, 42, 42, 30\
*Old*: "male", "female", "male", "female", "male", "female", 'male', 'female', 'male', 'female'\
*Perecentage*: 0.5, 0.5, 0.5, 0.7, 0.8, 0.2, 0.3, 0.5, 0.8, 0.4\

Here is a sample image from the datasets created.

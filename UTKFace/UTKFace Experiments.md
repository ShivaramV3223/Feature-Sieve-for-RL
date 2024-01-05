# Feature Sieve Regression Experiments
## Dataset: UTKFace

### Dataset Infos:
Dataset consists of total of 23705 datapoints.\
Divided into:
1) Training Dataset of 14223 datapoints.
2) Validation Dataset of 4741 datapoints.
3) Test Datset of 4741 datapoints

To experiment on simplicity bias, we create biased datasets by biasing the dataset using the attribute gender.\
We have 7470 male datapoints and 6753 female datapoints. The biased datasets contain 5000 datapoints.

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

Here is a sample image from the datasets created, for more images look into the Datasets/Gen1 folder under UTKFace.
![Dataset distribution 0 from generator 1](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Datasets/Gen1/dataset_dist0.png)

#### Data Generation 2
This function also biases the same target with the same attribute gender. This generation function is more mathematically oriented than the first one.\
This functions takes in the following parameters:
1) **Percentage**: Percentage split of the male population in the dataset.
2) **Male Mean**: Mean value of the male distribution which will be drawn from the train distribution.
3) **Mean Var**: Variance for the male distribution to be drawn.
4) **Female Mean**: Mean Values of the female distribution to be drawn.
5) **Female Var**: Variance of the female distribution to be drawn.

Here are the values used to produce the 10 datasets
1) percents = 0.8, 0.5, 0.6, 0.3, 0.3, 0.7, 0.8, 0.8, 0.2, 0.6
2) male_means = 20, 20, 30, 20, 40, 40, 40, 30, 40, 20
3) female_means = 80, 60, 60, 60, 20, 20, 20, 80, 50, 50
4) male_vars = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
5) female_vars = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

Here is the sample distribution from the second method of dataset generation, for all the datasets look into the Datasets/Gen2 folder under UTKFace.
![Dataset Distribution 0 from second method](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Datasets/Gen2/dataset_dist0.png)


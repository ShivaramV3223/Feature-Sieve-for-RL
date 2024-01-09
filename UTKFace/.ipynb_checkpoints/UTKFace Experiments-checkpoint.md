# Feature Sieve Regression Experiments
## Dataset: UTKFace

### Dataset Infos:
The dataset consists of a total of 23705 data points.\
Divided into:
1) Training Dataset of 14223 data points.
2) Validation Dataset of 4741 data points.
3) Test Datset of 4741 datapoints

To experiment on simplicity bias, we create biased datasets by biasing the dataset using the attribute gender.\
We have 7470 male data points and 6753 female data points. The biased datasets contain 5000 data points.

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

Here is a sample image from the datasets created, for more images look into the Datasets/Gen1 folder under UTKFace.\
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

Here is the sample distribution from the second method of dataset generation, for all the datasets look into the Datasets/Gen2 folder under UTKFace.\
![Dataset Distribution 0 from second method](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Datasets/Gen2/dataset_dist0.png)

### Models
1) CNN: A Basic resnet34 architecture for regression task
2) Feature Sieve Models:
   - All the feature sieve models have the same architecture. The main network consists of resent34 and the aux net consists of the first 2 Convolutional blocks of the resnet 34 architecture followed by average pooling and a Fully connected layer.
   - There are three Feature Sieve Models, each differ from the forgetting loss used.\
         1) *Feature Sieve Margin model*: This model uses a Margin forgetting loss as loss = min(-margin, -MSELOSS(aux_output, ground truth))\
         2) *Feature Sieve Cross Entropy model*: This model has its aux network as a classification task and bins the outputs into a few classes of the range. The model uses Cross Entropy loss as the forgetting loss as forget loss = cross entropy(aux_outputs, ones_like(aux_outputs) / num_bins).\
         3) *Feature Sieve Ordinal Model*: This model uses ordinal regression for training the aux layer. Every value in the range is converted into an ordinal label, the aux network is trained using a multi-label loss function to train the aux network. The forget loss is given as follows, loss = sum(aux_output_logits).\

The results of the model on datasets from dataset gen 1:\
![Results on the Datasets Gen 1](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Outputs/Test_Losses_Gen1.png)
The results of the model on datasets from dataset gen 2:\
![Results on the Datasets Gen 2](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Outputs/Test_Losses_Gen2.png)

Further here is the comparison of a Bayesian Regression model which uses just gender as input and predicts the age and the CNN Model on the Dataset Gen 2.\
![Bayesian Loss vs CNN Loss Gen2](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/UTKFace/Outputs/Test_Losses_Bayesian.png)
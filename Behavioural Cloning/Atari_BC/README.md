# Behavioral Cloning Atari 
## Dataset: D4RL Atari
The dataset consists of 10,00,000 data points.\
The dataset is split as follows:
1) Training Dataset: 9,50,000
2) Validation Dataset: 25,000
3) Test Datase: 25,000

Each data point is made by stacking 4 frames each is converted to Grayscale and resized to (84, 84).

### Models
1) **CNN Model**: A simple CNN from the DQN paper which consists of 3 convolutional layers and 2 fully connected layers
2) **Feature Sieve**: The Feature Sieve model has its main network of the same architecture as the CNN model and the aux network consists of 3 fully connected layers.

 Two different architectures of the Feature Sieve model differ where the aux network is connected to the main network. The first model is connected to the 1st convolutional layer and the second model has its aux network connected to the 2nd convolutional layer of the main network.

 Feature Sieve Model with its aux network connected to the 1st convolutional layer performs well.

 Here are the results obtained for:
 1) Space invaders:\
    ![result_space_invaders](https://github.com/ShivaramV3223/Feature-Sieve-for-RL/blob/main/Behavioural%20Cloning/Atari_BC/Outputs/cnn_vs_fs.png)

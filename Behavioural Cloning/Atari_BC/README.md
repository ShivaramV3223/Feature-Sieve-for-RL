# Behavioral Cloning Atari SpaceInvaders
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

 Two different architectures of the Feature Sieve model differ where does the aux network is connected to the main network. The first model is connected to the 1dt convolutional layer and the second model has its aux network connected to the 2nd convolutional layer of the main network.

# Libraries
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)

# Data
data("BostonHousing")  # data in mlbench package
data <- BostonHousing
str(data)

data %<>% mutate_if(is.factor, as.numeric) # convert factor data type to numeric

# Neural Network Visualization -- build the ANN chart
n <- neuralnet(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,
               data = data, # use BostonHousing dataset
               hidden = c(10,5),  # make 2 hidden layers, 1st hidden layer has 10 neurons, and 2nd has 5 neurons
               linear.output = F,  # regression:linear.output=TRUE, classification:linear.output=FALSE
               lifesign = 'full', # how much the function will print during the calculation of the neural network. 'none', 'minimal' or 'full'.
               rep=1)  # the number of repetitions for the neural network's training.
plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')



# Matrix -- convert dataset from to matrix table
data <- as.matrix(data)
dimnames(data) <- NULL

# Partition -- split dataset into 2 parts: training (70%) and testing (30%)
set.seed(1234) # let dataset random select the training data and testing data
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))  # split dataset (70% for training and 30% for testing)
training <- data[ind==1,1:13]  # put trainging data into training dataset, only take column 1 to 13, column 14 is the target)
test <- data[ind==2, 1:13]   # # put trainging data into testing dataset, only take column 1 to 13, column 14 is the target)
trainingtarget <- data[ind==1, 14] # put trainging target data into training target dataset, only take column 14)
testtarget <- data[ind==2, 14]  # put testing target data into testing target dataset, only take column 14)

# Normalize
m <- colMeans(training) # an inbuilt R function that calculates the means of each column of a matrix or array.
s <- apply(training, 2, sd) # apply(X, MARGIN, FUN, ...) -- X is dataset, margin=1 works on rows, 2 works on coulumns. sd is standard deviation funciotn.
training <- scale(training, center = m, scale = s)  # normalize the Training data set
test <- scale(test, center = m, scale = s)  # normalize the testing  

# Create Model
model <- keras_model_sequential()
model %>% #1 -- use 1 hidden layer with 5 neurons, input with 13 neurons(13 variables- columns)
  layer_dense(units = 5, activation = 'relu', input_shape = c(13)) %>%  # Applies the rectified linear unit (relu) activation function.
  layer_dense(units = 1) # ouput 1 neuron

model %>% #2 -- use 2 hidden layer with 10 and 5 neurons, input with 13 neurons(13 variables- columns)
  layer_dense(units = 10, activation = 'relu', input_shape = c(13)) %>%  # Applies the rectified linear unit (relu) activation function.
  layer_dense(units=5, activation = 'relu') %>%
  layer_dense(units = 1) # ouput 1 neuron

model %>% # 3 -- use 3 layers
  layer_dense(units = 100, activation = 'relu', input_shape = c(13)) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dense(units = 1)

model %>% # 4 -- use 3 layers with dropout
  layer_dense(units = 100, activation = 'relu', input_shape = c(13)) %>%
  layer_dropout(rate=0.4) %>%  # 40% of 100 neurons will be dropped which means won't be used in trainging
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 1)

model %>% #5 -- use 4 layers with dropout
  layer_dense(units = 300, activation = 'relu', input_shape = c(13)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 150, activation = 'relu') %>%
  layer_dropout(rate=0.35) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse',  # Mean square error
                  optimizer = 'rmsprop',  #Kares optimizer
                  metrics = 'mae')  # Mean absolute error

# Fit Model -- To chart the difference of training model prediction with real target value (mdv)
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

# Evaluate  -- Testing
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)

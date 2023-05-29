#Nonlinear Autoregressive Exogenous (NARX) Approach 
#Goal: Forecasting 20th hour Electricity Consumption using 18th and 19th hour values.  
#NN Architecture: 6 NNs with different combinations of input vectors, hidden layers, neurons, act. functions and learning rates. 

library(readxl)
library(dplyr)
library(MLmetrics)
#install.packages("devtools")
library(devtools)
#install_github("bips-hb/neuralnet")
library(neuralnet)

#dataset
data<- read_excel(" ")
head(data)

#train-test split
train_data<- data[1:380, 2:4]
test_data<- data[381:470, 2:4]

#col names-> 
#20th hour variables: X1=t-1, X2=t-2, X3=t-3, X4=t-4, X5=t-5, X6=t-6, X7=t-7, y=output
#Exogneous Variables: EX1= electricity consumption of the previous day for the 18th hour 
#                     EX2= electricity consumption of the previous day for the 19th hour 

#Chosen Input columns in the I/O Matrices: 
#                                          EX1 + EX2 
#                                          t-1 level + EX1 + EX2
#                                          t-1 level + t-7 + EX1 + EX2
#                                          t-4 level + EX1 + EX2
#                                          t-4 level + t-7 level + EX1 + EX2
#                                          t-7 level + EX1 + EX2

#Chosen Output column for all I/O Matrices: 
#                                          t value in the 20th hour 

#Exogenous var matrix
#training
exo_matrix_train<- cbind(train_data[, 1], train_data[, 2])
colnames(exo_matrix_train)<- c("EX1", "EX2")
#testing
exo_matrix_test<- cbind(train_data[, 1], train_data[, 2])
colnames(exo_matrix_test)<- c("EX1", "EX2")

#Time-Delayed input vectors and their I/O Matrices
create_io_matrix <- function(data, input_delay) {
  #X: time delayed values y: output/to be predicted value 
  X <- matrix(nrow = nrow(data) - input_delay, ncol = input_delay)
  y <- data[(input_delay + 1):nrow(data), 1]
  for (i in (input_delay + 1):nrow(data)) {
    X[i - input_delay, ] <- as.vector(t(data[(i - input_delay):(i - 1), ]))
  }
  
  return(list(X = X, y = y)) #list of X as a matrix with t-n values and y as a vector the relevant output 
}

input_delays <- c(1, 2, 3, 4) #time delays up to t-4 

#I/O matrices for TRAINING
io_matrices_train <- lapply(input_delays, function(delay) create_io_matrix(train_data[, 3], delay))
io_matrices_train
#Extracting and naming relevant training I/O matrices at t-n level 
#t-1 
colnames(io_matrices_train[[1]]$X)<-c("X1")
colnames(io_matrices_train[[1]]$y)<-c("y")
io_matrices_train[[1]]
#t-4, t-3, t-2, t-1
colnames(io_matrices_train[[4]]$X)<-c("X4", "X3", "X2", "X1")
colnames(io_matrices_train[[4]]$y)<-c("y")
io_matrices_train[[4]]

#I/O matrices for TESTING
io_matrices_test <- lapply(input_delays, function(delay) create_io_matrix(test_data[, 3], delay))
io_matrices_test
#Extracting relevant training I/O matrice at t-n level 
#t-1 level
colnames(io_matrices_test[[1]]$X)<-c("X1")
colnames(io_matrices_test[[1]]$y)<-c("Output")
io_matrices_test[[1]]
#t-4, t-3, t-2, t-1 level
colnames(io_matrices_test[[4]]$X)<-c("X4", "X3", "X2", "X1")
colnames(io_matrices_test[[4]]$y)<-c("Output")
io_matrices_test[[4]]

#loading t-7 to identify the influence of t-7 values on the forecast
t_7_train<- bind_cols(previous7= lag(train_data[, 3], 7),
                      previous6= lag(train_data[, 3], 6),
                      previous5= lag(train_data[, 3], 5),
                      previous4= lag(train_data[, 3], 4),
                      previous3= lag(train_data[, 3], 3),
                      previous2= lag(train_data[, 3], 2),
                      previous1= lag(train_data[, 3], 1),
                      pred= train_data[, 3])
colnames(t_7_train)<- c("X7", "X6", "X5", "X4", "X3", "X2", "X1", "y")
t_7_train
t_7_test<- bind_cols(previous7= lag(test_data[, 3], 7),
                     previous6= lag(test_data[, 3], 6),
                     previous5= lag(test_data[, 3], 5),
                     previous4= lag(test_data[, 3], 4),
                     previous3= lag(test_data[, 3], 3),
                     previous2= lag(test_data[, 3], 2),
                     previous1= lag(test_data[, 3], 1),
                     pred= test_data[, 3])
colnames(t_7_test)<- c("X7", "X6", "X5", "X4", "X3", "X2", "X1", "y")
t_7_test
#removing the NA values in both 
t_7_train<- t_7_train[complete.cases(t_7_train), ]
t_7_train
t_7_test<- t_7_test[complete.cases(t_7_test), ]
t_7_test

#extracting relevant data from t-7 level matrix 
#t-7 only (value 1 week before)
only_t_7_train<- cbind(t_7_train[, 1], t_7_train[, 8])
only_t_7_train
only_t_7_test<- cbind(t_7_test[, 1], t_7_test[, 8])
only_t_7_test
#t-1 with t-7 
t_1_with_t_7_train<- cbind(t_7_train[, 1], t_7_train[, 7], t_7_train[, 8])
t_1_with_t_7_train
t_1_with_t_7_test<- cbind(t_7_test[, 1], t_7_test[, 7], t_7_test[, 8])
t_1_with_t_7_test
#t-4, t-3, t-2, t-1 with t-7 
t_4_with_t_7_train<- cbind(t_7_train[, 1], t_7_train[, 4], t_7_train[, 5], t_7_train[, 6], t_7_train[, 7], t_7_train[, 8])
t_4_with_t_7_train
t_4_with_t_7_test<- cbind(t_7_test[, 1], t_7_test[, 4], t_7_test[, 5], t_7_test[, 6], t_7_test[, 7], t_7_test[, 8])
t_4_with_t_7_test

#NORMALIZING all I/O Matrices 
# Min-max normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

#DENORMALIZING NN's Outputs 
denormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

#Normalize the exo var train and test matrix
exo_matrix_train_norm<- apply(exo_matrix_train, 2, normalize)
exo_matrix_test_norm<- apply(exo_matrix_test, 2, normalize)

# Normalize up to t-4 level I/O training matrices in the list
io_matrices_train_norm<- io_matrices_train
for (i in 1:length(io_matrices_train)) {
  # Normalize inputs
  io_matrices_train_norm[[i]]$X <- apply(io_matrices_train[[i]]$X, 2, normalize)
  # Normalize outputs
  io_matrices_train_norm[[i]]$y <- normalize(io_matrices_train[[i]]$y)
}

# Normalize up to t-4 level I/O testing matrices in the list
io_matrices_test_norm<- io_matrices_test
for (i in 1:length(io_matrices_test)) {
  # Normalize inputs
  io_matrices_test_norm[[i]]$X <- apply(io_matrices_test[[i]]$X, 2, normalize)
  # Normalize outputs
  io_matrices_test_norm[[i]]$y <- normalize(io_matrices_test[[i]]$y)
}

#Normalizing the t-7 level training and testing matrices
only_t_7_train_matrix_norm<- apply(only_t_7_train, 2, normalize)
only_t_7_train_matrix_norm
only_t_7_test_matrix_norm<- apply(only_t_7_test, 2, normalize)
only_t_7_test_matrix_norm

#Normalizing the t-n level + t-7 level training and testing matrices 
#t-1 and t-7 norm
t_1_with_t_7_train_norm<- apply(t_1_with_t_7_train, 2, normalize)
t_1_with_t_7_train_norm
t_1_with_t_7_test_norm<- apply(t_1_with_t_7_test, 2, normalize)
t_1_with_t_7_test_norm

#t-4, t-3, t-2, t-1 and t-7 norm
t_4_with_t_7_train_norm<- apply(t_4_with_t_7_train, 2, normalize)
t_4_with_t_7_train_norm
t_4_with_t_7_test_norm<- apply(t_4_with_t_7_test, 2, normalize)
t_4_with_t_7_test_norm

#Training, testing and evaluating MLP models with various input vectors and internal structures.

#defining Symmetric MAPE index  
smape <- function(actual, forecast) {
  n <- length(actual)
  smape <- (100 / n) * sum(2 * abs(forecast - actual) / (abs(forecast) + abs(actual)))
  return(smape)
}

#I/O Matrix 1 for NN 1- 20th hour value based on the 18th and 19th hours of previous day
#Matrix 1 train 
io_matrix_1_train<- cbind(exo_matrix_train_norm[1:379, ], io_matrices_train_norm[[1]]$y)
io_matrix_1_train
#Matrix 1 test
io_matrix_1_test<- exo_matrix_test_norm[1:89, ]
io_matrix_1_test
#NN1 
mlp1<-neuralnet(y ~ EX1 + EX2, data= io_matrix_1_train, hidden = 10, act.fct= "tanh", linear.output = TRUE)
plot(mlp1)
#Testing 
y_test1_org<- as.matrix(io_matrices_test[[1]]$y) #y test valuesin original scale 
y_test1_org
y_pred1 <- predict(mlp1, io_matrix_1_test)
y_pred1
#denormalizing the normalized NN's output 
t1_output_min<- min(io_matrices_test[[1]]$y)
t1_output_max<- max((io_matrices_test[[1]]$y))
y_pred_org1<- denormalize(y_pred1,  t1_output_min, t1_output_max)
y_pred_org1
#performance evaluation
RMSE(y_test1_org, y_pred_org1)
MAE_NN1<- mean(abs(y_test1_org- y_pred_org1))
MAE_NN1
MAPE_NN1<- mean(abs((y_test1_org- y_pred_org1) / (y_test1_org))) * 100 
MAPE_NN1
SMAPE_NN1<- smape(y_test1_org, y_pred_org1)
SMAPE_NN1

#I/O Matrix 2 for NN 2- 20th hour value based on the 18th and 19th hours of previous day + t-1 level of 20th hour 
#Matrix 2 train 
io_matrix_2_train<- cbind(exo_matrix_train_norm[1:379, ], io_matrices_train_norm[[1]]$X, io_matrices_train_norm[[1]]$y)
io_matrix_2_train
#Matrix 2 test
io_matrix_2_test<- cbind(exo_matrix_test_norm[1:89, ], io_matrices_test_norm[[1]]$X)
io_matrix_2_test
#NN2
mlp2<-neuralnet(y ~ EX1 + EX2 + X1, data= io_matrix_2_train, hidden = c(3,2), act.fct= "relu", linear.output = FALSE)
plot(mlp2)
#Testing 
y_test1_org #y test valuesin original scale 
y_pred2 <- predict(mlp2, io_matrix_2_test)
y_pred2
#denormalizing the normalized NN's output 
y_pred_org2<- denormalize(y_pred2,  t1_output_min, t1_output_max)
y_pred_org2
#performance evaluation
RMSE(y_test1_org, y_pred_org2)
MAE_NN2<- mean(abs(y_test1_org- y_pred_org2))
MAE_NN2
MAPE_NN2<- mean(abs((y_test1_org- y_pred_org2) / (y_test1_org))) * 100 
MAPE_NN2
SMAPE_NN2<- smape(y_test1_org, y_pred_org2)
SMAPE_NN2

#I/O Matrix 3 for NN 3- 20th hour value based on the 18th and 19th hours of previous day + t-1 level + t-7 level of 20th hour 
#Matrix 3 train 
io_matrix_3_train<- cbind(exo_matrix_train_norm[1:373, ], t_1_with_t_7_train_norm)
io_matrix_3_train
#Matrix 3 test
io_matrix_3_test<- cbind(exo_matrix_test_norm[1:83, ], t_1_with_t_7_test_norm[, 1:2])
io_matrix_3_test
#NN3
mlp3<-neuralnet(y ~ EX1 + EX2 + X7 + X1, data= io_matrix_3_train, hidden = c(8,5), act.fct= "logistic", linear.output = FALSE)
plot(mlp3)
#Testing 
y_test7_org<- t_1_with_t_7_test[, 3]#y test valuesin original scale 
y_pred3 <- predict(mlp3, io_matrix_3_test)
y_pred3
#denormalizing the normalized NN's output 
t7_output_min<- min(t_7_test[, 8])
t7_output_max<- max(t_7_test[, 8])
y_pred_org3<- denormalize(y_pred3,  t7_output_min, t7_output_max)
y_pred_org3
#performance evaluation
RMSE(y_test7_org, y_pred_org3)
MAE_NN3<- mean(abs(y_test7_org- y_pred_org3))
MAE_NN3
MAPE_NN3<- mean(abs((y_test7_org- y_pred_org3) / (y_test7_org))) * 100 
MAPE_NN3
SMAPE_NN3<- smape(y_test7_org, y_pred_org3)
SMAPE_NN3

#I/O Matrix 4 for NN 4- 20th hour value based on the 18th and 19th of previous day + t-4 value of 20th hour 
#Matrix 4 train 
io_matrix_4_train<- cbind(exo_matrix_train_norm[1:376, ], io_matrices_train_norm[[4]]$X, io_matrices_train_norm[[4]]$y)
io_matrix_4_train
#Matrix 4 test
io_matrix_4_test<- cbind(exo_matrix_test_norm[1:86, ], io_matrices_test_norm[[4]]$X)
io_matrix_4_test
#NN4
mlp4<-neuralnet(y ~ EX1 + EX2 + X4 + X3 + X2 + X1, data= io_matrix_4_train, hidden = 8 ,act.fct= "tanh", linear.output = TRUE)
plot(mlp4)
#Testing 
y_test4_org<- as.matrix(io_matrices_test[[4]]$y) #y test valuesin original scale 
y_test4_org
y_pred4 <- predict(mlp4, io_matrix_4_test)
y_pred4
#denormalizing the normalized NN's output 
t4_output_min<- min(io_matrices_test[[4]]$y)
t4_output_max<- max((io_matrices_test[[4]]$y))
y_pred_org4<- denormalize(y_pred4,  t4_output_min, t4_output_max)
y_pred_org4
#performance evaluation
RMSE(y_test4_org, y_pred_org4)
MAE_NN4<- mean(abs(y_test4_org- y_pred_org4))
MAE_NN4
MAPE_NN4<- mean(abs((y_test4_org- y_pred_org4) / (y_test4_org))) * 100 
MAPE_NN4
SMAPE_NN4<- smape(y_test4_org, y_pred_org4)
SMAPE_NN4

#I/O Matrix 5 for NN 5- 20th hour value based on the 18th and 19th of previous day + t-4 value of 20th hour + t-7 level of 20th hour
#Matrix 5 train 
io_matrix_5_train<- cbind(exo_matrix_train_norm[1:373, ], t_4_with_t_7_train_norm)
io_matrix_5_train
#Matrix 5 test
io_matrix_5_test<- cbind(exo_matrix_test_norm[1:83, ], t_4_with_t_7_test_norm[, 1:5])
head(io_matrix_5_test)
#NN5
mlp5<-neuralnet(y ~ EX1 + EX2 + X7+ X4 + X3 + X2 + X1, data= io_matrix_5_train, hidden = c(2, 1), act.fct= "relu", linear.output = FALSE)
plot(mlp5)
#Testing 
y_test7_org
y_pred5 <- predict(mlp5, io_matrix_5_test)
y_pred5
#denormalizing the normalized NN's output 
y_pred_org5<- denormalize(y_pred5,  t7_output_min, t7_output_max)
y_pred_org5
#performance evaluation
RMSE(y_test7_org, y_pred_org5)
MAE_NN5<- mean(abs(y_test7_org- y_pred_org5))
MAE_NN5
MAPE_NN5<- mean(abs((y_test7_org- y_pred_org5) / (y_test7_org))) * 100 
MAPE_NN5
SMAPE_NN5<- smape(y_test7_org, y_pred_org5)
SMAPE_NN5

#I/O Matrix 6 for NN 6- 20th hour value based on the 18th and 19th of previous day + t-7 level of 20th hour
#Matrix 6 train 
io_matrix_6_train<- cbind(exo_matrix_train_norm[1:373, ], only_t_7_train_matrix_norm)
io_matrix_6_train
#Matrix 6 test
io_matrix_6_test<- cbind(exo_matrix_test_norm[1:83, ], only_t_7_test_matrix_norm[, 1])
colnames(io_matrix_6_test)<- c("EX1", "EX2", "X7")
head(io_matrix_6_test)
#NN6
mlp6<-neuralnet(y ~ EX1 + EX2 + X7, data= io_matrix_6_train, hidden = 12, act.fct= "logistic", linear.output = FALSE, learningrate= 0.01)
plot(mlp6)
#Testing 
y_test7_org
y_pred6 <- predict(mlp6, io_matrix_6_test)
y_pred6
#denormalizing the normalized NN's output 
y_pred_org6<- denormalize(y_pred6,  t7_output_min, t7_output_max)
y_pred_org6
#performance evaluation
RMSE(y_test7_org, y_pred_org6)
MAE_NN6<- mean(abs(y_test7_org- y_pred_org6))
MAE_NN6
MAPE_NN6<- mean(abs((y_test7_org- y_pred_org6) / (y_test7_org))) * 100 
MAPE_NN6
SMAPE_NN6<- smape(y_test7_org, y_pred_org6)
SMAPE_NN6
  

#MLP 5 of the NARX Approach is the best network since the statistical indices are at their lowest.
#Further, this model's structure is comparatively simple with fewer weights. 

#Graphical Comparison 
x = 1:length(y_test7_org)
plot(x, y_test7_org, col = "red", type = "l", lwd=2,
     main = "Electricity Consumption Forecast for 20:00 Hour", 
     ylab = "Consumption in kWh")
lines(x, y_pred_org5, col = "blue", lwd=2)
legend("topright",  legend = c("original amount of consumption", "forecasted amount of consumption"), 
       fill = c("red", "blue"))
grid() 

#Tabular Comparison 
cleanoutput <- cbind(y_test7_org, y_pred_org5)
colnames(cleanoutput)<- c("Actual", "Forecasted")
cleanoutput





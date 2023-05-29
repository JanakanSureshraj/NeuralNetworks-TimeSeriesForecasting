#Autoregressive (AR) Approach
#Goal: Forecasting 20th hour Electricity Consumption based on t-n level values.  
#NN Architecture: 13 NNs with different combinations of input vectors, hidden layers, neurons, act. functions and learning rates. 

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
train_data<- data[1:380, 4]
test_data<- data[381:470, 4]

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

#col names: X1=t-1, X2=t-2, X3=t-3, X4=t-4, X5=t-5, X6=t-6, X7=t-7, y=output

#I/O matrices for TRAINING
io_matrices_train <- lapply(input_delays, function(delay) create_io_matrix(train_data, delay))
io_matrices_train
#Extracting ana naming relevant training I/O matrices at t-n level 
#t-1 
colnames(io_matrices_train[[1]]$X)<-c("t-1")
colnames(io_matrices_train[[1]]$y)<-c("output")
io_matrices_train[[1]]
#t-2, t-1 
colnames(io_matrices_train[[2]]$X)<-c("t-2", "t-1")
colnames(io_matrices_train[[2]]$y)<-c("Output")
io_matrices_train[[2]]
#t-3, t-2, t-1 
colnames(io_matrices_train[[3]]$X)<-c("t-3", "t-2", "t-1")
colnames(io_matrices_train[[3]]$y)<-c("Output")
io_matrices_train[[3]]
#t-4, t-3, t-2, t-1
colnames(io_matrices_train[[4]]$X)<-c("t-4", "t-3", "t-2", "t-1")
colnames(io_matrices_train[[4]]$y)<-c("Output")
io_matrices_train[[4]]

#I/O matrices for TESTING
io_matrices_test <- lapply(input_delays, function(delay) create_io_matrix(test_data, delay))
io_matrices_test
#Extracting relevant training I/O matrice at t-n level 
#t-1 level
colnames(io_matrices_test[[1]]$X)<-c("t-1")
colnames(io_matrices_test[[1]]$y)<-c("Output")
io_matrices_test[[1]]
#t-2, t-1 level
colnames(io_matrices_test[[2]]$X)<-c("t-2", "t-1")
colnames(io_matrices_test[[2]]$y)<-c("Output")
io_matrices_test[[2]]
#t-3, t-2, t-1 level
colnames(io_matrices_test[[3]]$X)<-c("t-3", "t-2", "t-1")
colnames(io_matrices_test[[3]]$y)<-c("Output")
io_matrices_test[[3]]
#t-4, t-3, t-2, t-1 level
colnames(io_matrices_test[[4]]$X)<-c("t-4", "t-3", "t-2", "t-1")
colnames(io_matrices_test[[4]]$y)<-c("Output")
io_matrices_test[[4]]

#loading t-7 to identify the influence of t-7 values on the forecast
t_7_train<- bind_cols(previous7= lag(train_data, 7),
                      previous6= lag(train_data, 6),
                      previous5= lag(train_data, 5),
                      previous4= lag(train_data, 4),
                      previous3= lag(train_data, 3),
                      previous2= lag(train_data, 2),
                      previous1= lag(train_data, 1),
                      pred= train_data)
colnames(t_7_train)<- c("X7", "X6", "X5", "X4", "X3", "X2", "X1", "y")
t_7_train

t_7_test<- bind_cols(previous7= lag(test_data, 7),
                     previous6= lag(test_data, 6),
                     previous5= lag(test_data, 5),
                     previous4= lag(test_data, 4),
                     previous3= lag(test_data, 3),
                     previous2= lag(test_data, 2),
                     previous1= lag(test_data, 1),
                     pred= test_data)
colnames(t_7_test)<- c("X7", "X6", "X5", "X4", "X3", "X2", "X1", "y")
t_7_test

#removing the NA values in both 
t_7_train<- t_7_train[complete.cases(t_7_train), ]
t_7_train
t_7_test<- t_7_test[complete.cases(t_7_test), ]
t_7_test


#Extracting relevant data from t-7 level matrix 

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

#t-2, t-1 with t-7 
t_2_with_t_7_train<- cbind(t_7_train[, 1], t_7_train[, 6], t_7_train[, 7], t_7_train[, 8])
t_2_with_t_7_train
t_2_with_t_7_test<- cbind(t_7_test[, 1], t_7_test[, 6], t_7_test[, 7], t_7_test[, 8])
t_2_with_t_7_test

#t-3, t-2, t-1 with t-7 
t_3_with_t_7_train<- cbind(t_7_train[, 1], t_7_train[, 5], t_7_train[, 6], t_7_train[, 7], t_7_train[, 8])
t_3_with_t_7_train
t_3_with_t_7_test<- cbind(t_7_test[, 1], t_7_test[, 5], t_7_test[, 6], t_7_test[, 7], t_7_test[, 8])
t_3_with_t_7_test

#T-4, t-3, t-2, t-1 with t-7 
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

#t-2, t-1 and t-7 norm
t_2_with_t_7_train_norm<- apply(t_2_with_t_7_train, 2, normalize)
t_2_with_t_7_train_norm
t_2_with_t_7_test_norm<- apply(t_2_with_t_7_test, 2, normalize)
t_2_with_t_7_test_norm

#t-3, t-2, t-1 and t-7 norm
t_3_with_t_7_train_norm<- apply(t_3_with_t_7_train, 2, normalize)
t_3_with_t_7_train_norm
t_3_with_t_7_test_norm<- apply(t_3_with_t_7_test, 2, normalize)
t_3_with_t_7_test_norm

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

#<--MODEL 1 for t-1-->
#training
io_matrices_train_norm_t1<- as.matrix(cbind(io_matrices_train_norm[[1]]$X, io_matrices_train_norm[[1]]$y))
colnames(io_matrices_train_norm_t1)<- c("X1", "y")
mlp1<-neuralnet(y ~ X1, data= io_matrices_train_norm_t1, hidden = 10, linear.output = TRUE)
plot(mlp1)
#testing 
io_matrices_test_norm_t1<- as.matrix(cbind(io_matrices_test_norm[[1]]$X, io_matrices_test_norm[[1]]$y))
X_test1<- io_matrices_test_norm_t1[, 1]
y_test1<- io_matrices_test_norm_t1[, 2]
y_test1_org<- as.matrix(io_matrices_test[[1]]$y) #y test valuesin original scale 
y_test1_org
y_pred1 <- predict(mlp1, data.frame(X_test1))
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

#<--MODEL 2 for t-2-->
#training
io_matrices_train_norm_t2<- as.matrix(cbind(io_matrices_train_norm[[2]]$X, io_matrices_train_norm[[2]]$y))
colnames(io_matrices_train_norm_t2)<- c("X2","X1", "y")
mlp2<-neuralnet(y ~ X2+X1, data= io_matrices_train_norm_t2, hidden = c(3,2),act.fct= "relu", linear.output = FALSE)
plot(mlp2)
#testing 
io_matrices_test_norm_t2<- as.matrix(cbind(io_matrices_test_norm[[2]]$X, io_matrices_test_norm[[2]]$y))
X_test2<- io_matrices_test_norm_t2[, 1:2]
y_test2<- io_matrices_test_norm_t2[, 3]
y_test2_org<- as.matrix(io_matrices_test[[2]]$y) #y test valuesin original scale 
y_test2_org
y_pred2 <- predict(mlp2, data.frame(X_test2))
y_pred2
#denormalizing the normalized NN's output 
t2_output_min<- min(io_matrices_test[[2]]$y)
t2_output_max<- max((io_matrices_test[[2]]$y))
y_pred_org2<- denormalize(y_pred2,  t2_output_min, t2_output_max)
y_pred_org2
#performance evaluation
RMSE(y_test2_org, y_pred_org2)
MAE_NN2<- mean(abs(y_test2_org- y_pred_org2))
MAE_NN2
MAPE_NN2<- mean(abs((y_test2_org - y_pred_org2) / (y_test2_org)))*100
MAPE_NN2
SMAPE_NN2<- smape(y_test2_org, y_pred_org2)
SMAPE_NN2

#<--MODEL 3 for t-3-->
#training
io_matrices_train_norm_t3<- as.matrix(cbind(io_matrices_train_norm[[3]]$X, io_matrices_train_norm[[3]]$y))
colnames(io_matrices_train_norm_t3)<- c("X3","X2","X1", "y")
mlp3<-neuralnet(y ~ X3+X2+X1, data= io_matrices_train_norm_t3, hidden = c(8, 6),act.fct= "logistic", linear.output = TRUE, learningrate= 0.1)
plot(mlp3)
#testing 
io_matrices_test_norm_t3<- as.matrix(cbind(io_matrices_test_norm[[3]]$X, io_matrices_test_norm[[3]]$y))
X_test3<- io_matrices_test_norm_t3[, 1:3]
y_test3<- io_matrices_test_norm_t3[, 4]
y_test3_org<- as.matrix(io_matrices_test[[3]]$y) #y test values in original scale 
y_test3_org
y_pred3 <- predict(mlp3, data.frame(X_test3))
y_pred3
#denormalizing the normalized NN's output 
t3_output_min<- min(io_matrices_test[[3]]$y)
t3_output_max<- max((io_matrices_test[[3]]$y))
y_pred_org3<- denormalize(y_pred3,  t3_output_min, t3_output_max)
y_pred_org3
#performance evaluation
RMSE(y_test3_org, y_pred_org3)
MAE_NN3<- mean(abs(y_test3_org- y_pred_org3))
MAE_NN3
MAPE_NN3<- mean(abs((y_test3_org - y_pred_org3) / (y_test3_org))) * 100 
MAPE_NN3
SMAPE_NN3<- smape(y_test3_org, y_pred_org3)
SMAPE_NN3 

#<--MODEL 4 for t-4-->
#training
io_matrices_train_norm_t4<- as.matrix(cbind(io_matrices_train_norm[[4]]$X, io_matrices_train_norm[[4]]$y))
colnames(io_matrices_train_norm_t4)<- c("X4","X3","X2", "X1", "y")
mlp4<-neuralnet(y ~ X4+X3+X2+X1, data= io_matrices_train_norm_t4, hidden = c(5,3) ,act.fct= "tanh", linear.output = FALSE, learningrate= 0.2)
plot(mlp4)
#testing 
io_matrices_test_norm_t4<- as.matrix(cbind(io_matrices_test_norm[[4]]$X, io_matrices_test_norm[[4]]$y))
X_test4<- io_matrices_test_norm_t4[, 1:4]
y_test4<- io_matrices_test_norm_t4[, 5]
y_test4_org<- as.matrix(io_matrices_test[[4]]$y) #y test values in original scale 
y_test4_org
y_pred4 <- predict(mlp4, data.frame(X_test4))
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
MAPE_NN4<- mean(abs((y_test4_org - y_pred_org4) / (y_test4_org))) * 100 
MAPE_NN4
SMAPE_NN4<- smape(y_test4_org, y_pred_org4)
SMAPE_NN4 

#<--MODEL 5 for t-1-->
#training
mlp5<-neuralnet(y ~ X1, data= io_matrices_train_norm_t1, hidden = c(12, 10), act.fct= "relu", linear.output = FALSE, learningrate = 0.05, stepmax = 5000)
plot(mlp5)
#testing 
y_test1_org
y_pred5 <- predict(mlp5, data.frame(X_test1))
y_pred5
#denormalizing the normalized NN's output 
y_pred_org5<- denormalize(y_pred5,  t1_output_min, t1_output_max)
y_pred_org5
#performance evaluation
RMSE(y_test1_org, y_pred_org5)
MAE_NN5<- mean(abs(y_test1_org- y_pred_org5))
MAE_NN5
MAPE_NN5<- mean(abs((y_test1_org- y_pred_org5) / (y_test1_org))) * 100 #small constant 0.0001 as some elements contain 0-preventing from zero div error 
MAPE_NN5
SMAPE_NN5<- smape(y_test1_org, y_pred_org5)
SMAPE_NN5

#<--MODEL 6 for t-2-->
#training
mlp6<-neuralnet(y ~ X2+X1, data= io_matrices_train_norm_t2, hidden = 3, act.fct= "tanh", linear.output = FALSE, learningrate= 0.1)
plot(mlp6)
#testing 
y_test2_org
y_pred6 <- predict(mlp6, data.frame(X_test2))
y_pred6
#denormalizing the normalized NN's output 
y_pred_org6<- denormalize(y_pred6,  t2_output_min, t2_output_max)
y_pred_org6
#performance evaluation
RMSE(y_test2_org, y_pred_org6)
MAE_NN6<- mean(abs(y_test2_org- y_pred_org6))
MAE_NN6
MAPE_NN6<- mean(abs((y_test2_org - y_pred_org6) / (y_test2_org)))*100
MAPE_NN6
SMAPE_NN6<- smape(y_test2_org, y_pred_org6)
SMAPE_NN6

#<--MODEL 7 for t-3-->
#training
mlp7<-neuralnet(y ~ X3+X2+X1, data= io_matrices_train_norm_t3, hidden = 12, act.fct= "logistic", linear.output = FALSE)
plot(mlp7)
#testing 
y_test3_org
y_pred7 <- predict(mlp7, data.frame(X_test3))
y_pred7
#denormalizing the normalized NN's output 
y_pred_org7<- denormalize(y_pred7,  t3_output_min, t3_output_max)
y_pred_org7
#performance evaluation
RMSE(y_test3_org, y_pred_org7)
MAE_NN7<- mean(abs(y_test3_org- y_pred_org7))
MAE_NN7
MAPE_NN7<- mean(abs((y_test3_org - y_pred_org7) / (y_test3_org))) * 100 
MAPE_NN7
SMAPE_NN7<- smape(y_test3_org, y_pred_org7)
SMAPE_NN7 

#<--MODEL 8 for t-4-->
#training
mlp8<-neuralnet(y ~ X4+X3+X2+X1, data= io_matrices_train_norm_t4, hidden = c(4,2), act.fct= "tanh",linear.output = TRUE, learningrate= 0.1)
plot(mlp8)
#testing 
y_test4_org
y_pred8 <- predict(mlp8, data.frame(X_test4))
y_pred8
#denormalizing the normalized NN's output 
y_pred_org8<- denormalize(y_pred8,  t4_output_min, t4_output_max)
y_pred_org8
#performance evaluation
RMSE(y_test4_org, y_pred_org8)
MAE_NN8<- mean(abs(y_test4_org- y_pred_org8))
MAE_NN8
MAPE_NN8<- mean(abs((y_test4_org - y_pred_org8) / (y_test4_org))) * 100 
MAPE_NN8
SMAPE_NN8<- smape(y_test4_org, y_pred_org8)
SMAPE_NN8 

#<--MODEL 9 for only t-7-->
#training
only_t_7_train_matrix_norm
mlp9<-neuralnet(y ~ X7, data= only_t_7_train_matrix_norm, hidden = c(3, 2), act.fct = "logistic", linear.output = TRUE)
plot(mlp9)
#testing 
only_t_7_test_matrix_norm
X_test_only_t7<- only_t_7_test_matrix_norm[, 1]
y_test_t7_org<- as.matrix(t_7_test[ ,8]) #values for y in original scale 
y_test_t7_org
y_pred9 <- predict(mlp9, data.frame(X_test_only_t7))
y_pred9
#denormalizing the normalized NN's output 
t7_output_min<- min(t_7_test[, 8])
t7_output_max<- max(t_7_test[, 8])
y_pred_org9<- denormalize(y_pred9,  t7_output_min, t7_output_max)
y_pred_org9
#performance evaluation
RMSE(y_test_t7_org, y_pred_org9)
MAE_NN9<- mean(abs(y_test_t7_org- y_pred_org9))
MAE_NN9
MAPE_NN9<- mean(abs((y_test_t7_org- y_pred_org9) / (y_test_t7_org))) * 100 
MAPE_NN9
SMAPE_NN9<- smape(y_test_t7_org, y_pred_org9)
SMAPE_NN9

#<--MODEL 10 for t-1 with t-7-->
#training
mlp10<-neuralnet(y ~ X7+X1, data= t_1_with_t_7_train_norm, hidden = 5, act.fct = "relu", linear.output = FALSE)
plot(mlp10)
#testing 
X_test_t1_with_t7<- cbind(t_1_with_t_7_test_norm[, 1], t_1_with_t_7_test_norm[, 2])
X_test_t1_with_t7
y_pred10<- predict(mlp10, X_test_t1_with_t7)
y_pred10
#denormalizing the normalized NN's output 
y_pred_org10<- denormalize(y_pred10,  t7_output_min, t7_output_max)
y_pred_org10
#performance evaluation
RMSE(y_test_t7_org, y_pred_org10)
MAE_NN10<- mean(abs(y_test_t7_org- y_pred_org10))
MAE_NN10
MAPE_NN10<- mean(abs((y_test_t7_org- y_pred_org10) / (y_test_t7_org))) * 100 
MAPE_NN10
SMAPE_NN10<- smape(y_test_t7_org, y_pred_org10)
SMAPE_NN10

#<--MODEL 11 for t-2, t-1 with t-7-->
#training
mlp11<-neuralnet(y ~ X7+X2+X1, data= t_2_with_t_7_train_norm, hidden = 10, act.fct = "tanh", linear.output = TRUE)
plot(mlp11)
#testing 
X_test_t2_with_t7<- cbind(t_2_with_t_7_test_norm[, 1], t_2_with_t_7_test_norm[, 2], t_2_with_t_7_test_norm[, 3])
X_test_t2_with_t7
y_pred11<- predict(mlp11, X_test_t2_with_t7)
y_pred11
#denormalizing the normalized NN's output 
y_pred_org11<- denormalize(y_pred11,  t7_output_min, t7_output_max)
y_pred_org11
#performance evaluation
RMSE(y_test_t7_org, y_pred_org11)
MAE_NN11<- mean(abs(y_test_t7_org- y_pred_org11))
MAE_NN11
MAPE_NN11<- mean(abs((y_test_t7_org- y_pred_org11) / (y_test_t7_org))) * 100 
MAPE_NN11
SMAPE_NN11<- smape(y_test_t7_org, y_pred_org11)
SMAPE_NN11

#<--MODEL 12 for t-3, t-2, t-1 with t-7-->
#training
mlp12<-neuralnet(y ~ X7+X3+X2+X1, data= t_3_with_t_7_train_norm, hidden = c(5, 3), act.fct = "logistic", linear.output = FALSE)
plot(mlp12)
#testing 
X_test_t3_with_t7<- cbind(t_3_with_t_7_test_norm[, 1], t_3_with_t_7_test_norm[, 2], t_3_with_t_7_test_norm[, 3], t_3_with_t_7_test_norm[, 4])
X_test_t3_with_t7
y_pred12<- predict(mlp12, X_test_t3_with_t7)
y_pred12
#denormalizing the normalized NN's output 
y_pred_org12<- denormalize(y_pred12,  t7_output_min, t7_output_max)
y_pred_org12
#performance evaluation
RMSE(y_test_t7_org, y_pred_org12)
MAE_NN12<- mean(abs(y_test_t7_org- y_pred_org12))
MAE_NN12
MAPE_NN12<- mean(abs((y_test_t7_org- y_pred_org12) / (y_test_t7_org))) * 100 
MAPE_NN12
SMAPE_NN12<- smape(y_test_t7_org, y_pred_org12)
SMAPE_NN12


#<--MODEL 13 for t-4, t-3, t-2, t-1 with t-7-->
#training
mlp13<-neuralnet(y ~ X7+X4+X3+X2+X1, data= t_4_with_t_7_train_norm, hidden = c(3,2), act.fct = "relu", linear.output = TRUE, learningrate= 0.1)
plot(mlp13)
#testing 
X_test_t4_with_t7<- cbind(t_4_with_t_7_test_norm[, 1], t_4_with_t_7_test_norm[, 2], t_4_with_t_7_test_norm[, 3], t_4_with_t_7_test_norm[, 4], t_4_with_t_7_test_norm[, 5])
X_test_t4_with_t7
y_pred13<- predict(mlp13, X_test_t4_with_t7)
y_pred13
#denormalizing the normalized NN's output 
y_pred_org13<- denormalize(y_pred13,  t7_output_min, t7_output_max)
y_pred_org13
#performance evaluation
RMSE(y_test_t7_org, y_pred_org13)
MAE_NN13<- mean(abs(y_test_t7_org- y_pred_org13))
MAE_NN13
MAPE_NN13<- mean(abs((y_test_t7_org- y_pred_org13) / (y_test_t7_org))) * 100 
MAPE_NN13
SMAPE_NN13<- smape(y_test_t7_org, y_pred_org13)
SMAPE_NN13


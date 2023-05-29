# NeuralNetworks-TimeSeriesForecasting
Electricity consumption forecasting on electricity consumption data (in kWh) for a building at 20:00, 19:00 and 18:00 hours daily for the years 2018 and 2019.
Objective of this project is to use a multilayer neural network (MLP-NN) to predict the next step-ahead (i.e. next day) electricity consumption for the 20:00 hour case.						
		
AR_Approach.R- The 1st subtask, one-step-ahead forecasting of electricity consumption will utilise only the “autoregressive” (AR) approach (i.e. time-delayed values of the 20th hour attribute as input variables). Input vectors upto t-4 level have been computed. According to literature, the electricity consumption forecast depends also on the (t-7) (i.e. one week before) value of the load. Thus, the AR approach also investigates the influence of this specific time-delayed load to the forecasting performance of the NN models. Input/Output matrices by combining different t-n levels have been fed to a total of 13 Neural Networks and performances have been evaluated. 
	
NARX_Approach.R- The 2nd subtask, however, one-step-ahead forecasting of electricity consumption will utilise additional input vectors by including information from the 19th and 18th hour attributes. In that case, these NN models could be considered as “NARX” (nonlinear autoregressive exogenous) style models. Initial Input/Output matrices + exogenous variables (as columns) have been used to train 6 Neural Networks. 
A total of 19 Neural Networks for both the above approaches have been developed and one best model has been prefered based on statistical indices such as RMSE, MAE, MAPE and SMAPE. 

 

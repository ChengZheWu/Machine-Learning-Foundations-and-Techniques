# The daily revenue of the hotel booking company prediction

# Introduction
In this final project, I am going to be part of an exciting machine learning competition. A hotel booking company tries to predict the daily revenue of the company from the data of reservation. In particular, after a room reservation request is fulfilled (i.e. no cancellation), on the arrival date, the revenue of the request is the rate of the room (called ADR) multiplied by the number of days that the customer is going to stay in the room, and the daily revenue is the sum of all those fulfilled requests on the same day. The goal of the prediction is to accurately infer the future daily revenue of the company, where the daily revenue is quantized to 10 scales.

# Data Sets
The [data sets](https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/) are processed from the [Kaggle hotel booking demand data](https://www.kaggle.com/jessemostipak/hotel-booking-demand). There are 91531 pieces of data in training set with 33 features and 27859 pieces of data in testing set with 29 features.

# Data pre-processing
We simply clean the data with the following precedures:  
1. delete some useless data we thought
2. one hot enconding
3. replace "NaN" with zero
Finally, the dimension of training data and testing data are (88974, 931) and (27859, 931) respectively.

# Method & Results
We divide the task into 3 steps.
All the hyerparameters of models are the defaults of scikit-learn package. 
## 1. Cancellation Prediction
Model                     | Val Acc (%) | Time (s)  
:-------------------------|------------:|----------:
Random Forest Classifier  |**90.43**        |57.36         
Nearest Neibor Classifier |80.47        |122.14   
SVM Classifier (RBF)      |68.46        |5992.95
## 2. Adr Prediction
Model                     | Val MAE     | Time (s)  
:-------------------------|------------:|----------:
Random Forest Regressor   |**11.27**        |69.06       
Nearest Neibor Regressor  |19.52        |2.94   
SVM Regressor (RBF)       |31.57        |5365.73
## 3. Revenue Calculation
 $$ max\(min\(\lfloor\frac {1} {2}\rfloor, 9), 0) $$

Customer Churn Prediction
This project aims to predict customer churn for a banking institution using a neural network model. The model is built with TensorFlow and Keras and is optimized using Keras Tuner to find the best hyperparameters.

Dataset
The project uses the Churn_Modelling.csv dataset. This dataset contains information about bank customers, including their credit score, geography, gender, age, tenure, balance, number of products, and whether they have a credit card. The "Exited" column is the target variable, indicating whether a customer has churned (1) or not (0).

Installation
To run this project, you need to have Python and the following libraries installed:

Bash

pip install numpy pandas tensorflow scikit-learn matplotlib keras-tuner
Usage
Clone this repository to your local machine.
Make sure you have the Churn_Modelling.csv file in the same directory as the notebook.
Open the customer_churn_prediction.ipynb notebook in a Jupyter environment.
Run the cells in the notebook sequentially to see the data preprocessing, model training, and evaluation steps.
Model Architecture
The neural network model is a sequential model with the following layers:

Layer (type)	Output Shape	Param #
dense_4 (Dense)	(None, 480)	5,760
batch_normalization (BatchNormalization)	(None, 480)	1,920
dropout (Dropout)	(None, 480)	0
dense_5 (Dense)	(None, 320)	153,920
batch_normalization_1 (BatchNormalization)	(None, 320)	1,280
dropout_1 (Dropout)	(None, 320)	0
dense_6 (Dense)	(None, 480)	154,080
batch_normalization_2 (BatchNormalization)	(None, 480)	1,920
dropout_2 (Dropout)	(None, 480)	0
dense_7 (Dense)	(None, 1)	481

Export to Sheets
Total params: 319,361
Trainable params: 316,801
Non-trainable params: 2,560

The model uses the relu activation function in the hidden layers and the sigmoid activation function in the output layer for binary classification. It also incorporates BatchNormalization and Dropout to improve training stability and prevent overfitting. The best hyperparameters, found using Keras Tuner, are:

Number of Layers: 3
Units per Layer: [384, 320, 192]
Optimizer: rmsprop
Results
The model was trained for 30 epochs and achieved an accuracy of approximately 79.8% on the test set. The classification report below provides a more detailed look at the model's performance:

              precision    recall  f1-score   support

           0       0.93      0.81      0.86      1595
           1       0.50      0.77      0.61       405

    accuracy                           0.80      2000
   macro avg       0.72      0.79      0.73      2000
weighted avg       0.84      0.80      0.81      2000
The model shows good performance in identifying customers who have not churned (class 0), but it has lower precision for identifying customers who have churned (class 1). However, the recall for class 1 is good, which is important for this problem as we want to identify as many churning customers as possible.

Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

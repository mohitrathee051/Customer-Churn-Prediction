🤖 Customer Churn Prediction 🤖
Welcome to the Customer Churn Prediction project! This repository contains a Jupyter notebook that walks through the process of building a neural network to predict customer churn for a bank. churn.

✨ Key Features
Data Preprocessing & EDA: Cleans and prepares the Churn_Modelling.csv dataset for modeling and performs exploratory data analysis to uncover insights.
Hyperparameter Tuning: Uses Keras Tuner to find the optimal hyperparameters for the neural network, ensuring the best possible performance.
Neural Network Model: Builds a robust neural network using TensorFlow and Keras to classify customers as "churned" or "not churned."
Class Imbalance Handling: Addresses the class imbalance in the dataset to prevent the model from being biased towards the majority class.
Model Evaluation: Provides a detailed evaluation of the model's performance using metrics like accuracy, precision, recall, and F1-score.
🛠️ Technologies Used
Python
Pandas & NumPy: For data manipulation and numerical operations.
TensorFlow & Keras: For building and training the neural network.
Scikit-learn: For data preprocessing and evaluation.
Matplotlib: For data visualization.
Keras Tuner: For hyperparameter optimization.
🚀 How To Use
Clone the repository:
Bash

git clone https://github.com/your-username/customer-churn-prediction.git
Install the dependencies:
Bash

pip install numpy pandas tensorflow scikit-learn matplotlib keras-tuner
Download the dataset: Make sure you have the Churn_Modelling.csv file in the project directory.
Run the Jupyter Notebook: Open and run the customer_churn_prediction.ipynb notebook to see the complete workflow.
🧠 Model Architecture
The model is a sequential neural network with the following architecture, optimized by Keras Tuner:

Input Layer
Dense Layer: 480 neurons, ReLU activation
Batch Normalization & Dropout
Dense Layer: 320 neurons, ReLU activation
Batch Normalization & Dropout
Dense Layer: 480 neurons, ReLU activation
Batch Normalization & Dropout
Output Layer: 1 neuron, Sigmoid activation
This architecture, combined with L1 regularization, helps in creating a robust model that generalizes well to new data.

📊 Results
The model achieves an accuracy of approximately 79.8% on the test set. Here's a summary of the classification report:

Precision	Recall	F1-Score
Did Not Churn (0)	0.93	0.81	0.86
Churned (1)	0.50	0.77	0.61

Export to Sheets
The model is particularly good at identifying customers who are likely to churn (high recall for class 1), which is crucial for this business problem.

🤝 Contributing
Contributions are welcome! If you have any ideas for improvement or find any issues, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

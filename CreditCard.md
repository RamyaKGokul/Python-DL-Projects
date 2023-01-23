## Credit Card Fraud Detector
An implementation of a binary sequential neural network model for classifying the fraudulent and non-fraudulent credit card transactions using big dataset containing 284,807 transactions and achieved an accuracy over 99.8%. 

### Dataset creditcard.csv
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly imbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation.
Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for imbalanced classification.

### Data Description(Sample)
<img width="1641" alt="Screen Shot 2023-01-23 at 12 13 01 PM" src="https://user-images.githubusercontent.com/122699563/214106645-7f45a27f-6e8b-4675-bf23-b271784aeca8.png">

### Imbalanced data- visualization
<img width="492" alt="Screen Shot 2023-01-23 at 12 13 32 PM" src="https://user-images.githubusercontent.com/122699563/214106785-835903de-1a6d-4ca4-bf7e-331b930d199c.png">
 
### After data oversampling
<img width="511" alt="Screen Shot 2023-01-23 at 12 13 49 PM" src="https://user-images.githubusercontent.com/122699563/214106967-a0deeebe-ad03-49bc-b049-25bf916a3f40.png">

### Binary classifier architecture
<img width="574" alt="Screen Shot 2023-01-23 at 12 14 04 PM" src="https://user-images.githubusercontent.com/122699563/214107292-e6dc8223-2028-4825-9e15-fb1fd0f39e8f.png">

### Model Accuracy and Loss
<img width="1438" alt="Screen Shot 2023-01-23 at 12 14 24 PM" src="https://user-images.githubusercontent.com/122699563/214107351-98e854e6-82e1-4501-a0c8-42edfa5abdee.png">

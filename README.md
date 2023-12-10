# Predicting-Defaulting-Borrower

Project name: Home Credit Group. Loan Behavior Predictor. The goal of this project is to create a web service to predict credit applicants’ behavior. The data for this project contains various tables with historical applicant information. The main table is app_train, where current applicants’ data is provided. In addition to that there are tables from external sources that contain information about the applicant’s past behavior handling previous loans, credit card and cash behavior. The data is provided in several different tables: application, bureau, bureau balance, credit cards, pos cash, payments, and previous applications.

The Plan for handing data will be the following:

Analize all the columns of provided data.
Determine the aggregation strategy for each table with the purpose of joining the external sources data with the application table.
Perform EDA with the purpose of identifying the key features that has an impact on the potential defaulted behavior.
Create the list of Major features to use for modeling.
Perform various model training with the Major features, full joined features, and with the application original columns.
Identify the best performing model and perform hyperparameter tunning.
Test the model with the application sample.
Load the functioning model to the cloud.
In addition, this project aims to:

To analyze what were the potential criteria for rejecting the loan in the past.
Practice identifying opportunities for data analysis, raising hypothesis, and formulating research tasks.
Practice performing EDA, statistical inference, and prediction.
Practice visualizing data.
Practice machine learning modeling techniques.
Practicing deploying the model for production in Google Cloud Service.
Project files:
Main folder:
1. Project.ipynb – the main notebook where data is analyzed and joined, model selection, training, hyperparameter tunning takes place.
2. Eda-functions.py – main EDA functions that were used in the project.
3. Lists.py – contains most of the lists that were created for data aggregations purposes.
4. Transformation_functions.py – some additional data transformation functions.
5. Model_functions.py – modeling functions.
6. Main.py – Flask file for prediction model loading to GCP.
7. Model.pkl – saved trained model.
8. Requirements.txt – libraries that needed to be loaded to virtual environment.
9. App.yaml – loading instructor for gcloud.

Test folder
1. Test.py – testing file for online and offline testing. For offline testing please # comment the line in the code “app_url = https://loans-404215.lm.r.appspot.com/predict” and uncomment the “# app_url = http://127.0.0.1:8080/predict” – also, make sure that the local host address is the same at your machine – adjust if needed. (Test function is disabled for now to stop GCP charging for resourse using, but can be renewed upon request).
2. Joined data sample.csv - a sample file randomly created for the testing purposes.
3. Test data sample.csv - addition test data provided by the competition (for those that have the correct Target variable for test data) 

Reading notebook:

Data files should be accessed from this location: https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip
After downloading, store all the files in a newly created folder “Data”, which should be stored in the same folder with project files.

You are free to suggest any changed on how the project could be improved. The project is not bind by any license agreement.

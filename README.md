Receipt Matching Data Science Challenge

Given a set of receipt-transactions, this code will produce a trained model which can later be used to order transactions 
by likelihood of matching a receipt image.

Pre-requisites

1. Please change the current working directory to receipt_transaction_matching folder.
2. Change the path of config file to te appropriate path, at line 3 of config.py file.
3. Give the appropriate path to train file,test file,path to save model & results config.yaml present in config folder.
   I have given the paths of the files I used from the "dataset" folder in this project.
4. Install the packages in requirements.txt

Run the project
   
1. Run train_model/training_model.py to train & save model in given folder.   
2. To test model use test_model/testing_model.py. Give the path to the model you want to use for testing at line 7
   in testing_model.py. The files should be in ".csv" format.

Check the result   
1. Result will also be stored in ".csv" format, with the matched transaction at the top of the records in test file.  

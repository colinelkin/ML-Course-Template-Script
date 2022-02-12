"""STEP 1: Preliminary language-specific commands"""
import os
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, datasets
from sklearn.ensemble import RandomForestRegressor

"""STEP 2: Load the data"""
input_data, output_data = datasets.fetch_california_housing(return_X_y=True)

"""STEP 3: Shuffle the samples and split into train and test"""
[train_in, test_in, train_out, test_out] = model_selection.train_test_split(input_data, output_data, test_size=.2)

"""STEP 4: Determine the hyperparameters"""
model = RandomForestRegressor()

"""STEP 5: Train the model"""
model.fit(train_in, train_out)

"""STEP 6: Predict training outputs"""
pred_train_out = model.predict(train_in)

"""STEP 7: Evaluate the training data"""
eval_method = "Mean Squared Error"
train_score = metrics.mean_squared_error(train_out, pred_train_out)

"""STEP 8: Predict test outputs"""
pred_test_out = model.predict(test_in)
    
"""STEP 9: Get the testing score"""
test_score = metrics.mean_squared_error(test_out, pred_test_out)
    
"""STEP 10: Save evaluation results and outputs to a file"""
# training and testing results
results = np.array([["Training " + eval_method + " (%): ", 100 * train_score], ["Testing " + eval_method + " (%): ", 100 * test_score]])
results_file = pd.DataFrame(results)
# predicted values versus actual values on training data
train_compare = pd.DataFrame((np.transpose((np.vstack((pred_train_out,np.transpose(train_out)))))))
# predicted values versus actual values on testing data
test_compare = pd.DataFrame((np.transpose((np.vstack((pred_test_out,np.transpose(test_out)))))))

# filepath to "Saved Files" folder
savedir = "Saved Files" + os.sep
# export evaluation results
results_file.to_csv(savedir + eval_method + ".csv", index = False, header = False)
# export training outputs
train_compare.to_csv(savedir + "Training Outputs.csv", index = False, header = ["Predicted", "Actual"])
# export test outputs
test_compare.to_csv(savedir + "Test Outputs.csv", index = False, header = ["Predicted", "Actual"])

"""STEP 11: Display results to the console"""
for elt in results: print(*elt, sep="\n")
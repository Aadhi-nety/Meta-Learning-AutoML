# src/automl.py

import h2o
from h2o.estimators import H2OAutoML

# Initialize the H2O cluster
h2o.init()

# Load and process the dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = h2o.import_file(data_url)

# Define the response and feature variables
response = "class"
x = data.columns
x.remove(response)

# Split the dataset
train, valid = data.split_frame(ratios=[.8], seed=42)

# Run H2O AutoML
aml = H2OAutoML(max_runtime_secs=300, seed=42)
aml.train(x=x, y=response, training_frame=train, validation_frame=valid)

# Get the leaderboard
leaderboard = aml.leaderboard
print(leaderboard)

# Shutdown the H2O cluster
h2o.shutdown()

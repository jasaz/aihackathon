import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from google.cloud import storage
#import gcsfs

def predict_model(test_file_bucket_path, ohe_model_bucket_name, ohe_model_file_name, trained_model_bucket_name, trained_model_file_name):

  # Create a Cloud Storage client.
  client = storage.Client()

  # Create a gcsfs filesystem.
  #fs = gcsfs.GCSFileSystem(project=project_id)

  # Read the file from Cloud Storage.
  #input_data_df = pd.read_csv(fs.open(test_file_object))
  input_data_df = pd.read_csv(test_file_bucket_path)
  
  input_data_df = input_data_df.dropna()
  test_data_df = input_data_df.astype({"envId":'int',"value":'int',"SummaryId":'int',"descriptionID":'int', "storypoints":'int'})
  
  categorical_columns = ["envId","value", "SummaryId", "descriptionID"]

  # Get the OHE model from Cloud Storage.
  blob = client.bucket(ohe_model_bucket_name).blob(ohe_model_file_name)

  # Open the OHE model in binary mode.
  with open(blob.open("rb"), "rb") as f:
    encoder = pickle.load(f)
  

  # Transform the test data.
  test_data = encoder.transform(test_data_df[categorical_columns])
  test_encoded_df = pd.DataFrame(test_data.toarray())
    
  column_names = []
  for i in range(test_encoded_df.shape[1]):
    column_names.append('feature_' + str(i))

  test_encoded_df.columns = column_names

  # Get the Trained model from Cloud Storage.
  model_blob = client.bucket(trained_model_bucket_name).blob(trained_model_file_name)

  # Open the Trained model in binary mode.
  with open(model_blob.open("rb"), "rb") as f:
    model = pickle.load(f)
  
  #model = pickle.load(open(trained_model_name, "rb"))

  prediction = model.predict(test_encoded_df)
  return prediction

def main():
  test_file_bucket_path = ""
  ohe_model_bucket_name = ""
  ohe_model_file_name = ""
  trained_model_bucket_name = ""
  trained_model_file_name = ""
  
  prediction = predict_model(test_file_bucket_path, ohe_model_bucket_name, ohe_model_file_name, trained_model_bucket_name, trained_model_file_name)
  return prediction
  

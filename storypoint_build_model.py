import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def process_data(input_file_name, encoder_name):

  feature_columns = columns = ["envId","value", "SummaryId", "descriptionID"]
  label = ["storypoints"]
  categorical_columns = ["envId","value", "SummaryId", "descriptionID"]

  input_data_df = pd.read_csv(input_file_name)
  input_data_df = input_data_df.dropna()
  data_df = input_data_df.astype({"envId":'int',"value":'int',"SummaryId":'int',"descriptionID":'int', "storypoints":'int'})

  label_df = data_df[label]
  
  data_df_temp = data_df.drop('storypoints', axis =1)

  encoder = OneHotEncoder()
  data_encoded = encoder.fit_transform(data_df_temp[categorical_columns])
    
  with open(encoder_name, "wb") as f:
    pickle.dump(encoder, f)

  data_encoded_df = pd.DataFrame(data_encoded.toarray())
  data_encoded_df.columns

  column_names = []
  for i in range(data_encoded_df.shape[1]):
    column_names.append('feature_' + str(i))
  
  data_encoded_df.columns = column_names
  
  final_df = pd.concat([data_encoded_df, label_df], axis =1)

  return final_df


def train_model(processed_df, test_size, random_state, shuffle, model_name):
  
  label_df = processed_df["storypoints"]
  processed_feature_df = processed_df.drop('storypoints', axis =1)
  X_train, X_test, y_train, y_test = train_test_split(processed_feature_df, label_df, test_size=test_size, random_state=random_state, shuffle=shuffle)
  clf = SVC(kernel="linear", decision_function_shape="ovo")
  clf.fit(X_train, y_train)

  # Evaluate the classifier
  score = clf.score(X_test, y_test)
  print("Accuracy of the model is: ", score)

  # Save the model
  pickle.dump(clf, open(model_name, "wb"))
  print ("The model has been created")
  

final_df = process_data("input_file_path", "one_hot_encoder.pkl")
train_model(final_df, 0.25, 42, True, 'storypoint_model.pkl')

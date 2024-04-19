# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json, os
import re
from typing import Any, Optional
from uuid import UUID
import pandas as pd

from langchain.callbacks.base import BaseCallbackHandler
from datetime import datetime


import time
# import UUIDEncoder

class BaseLogCallback(BaseCallbackHandler):
    def __init__(self, output_csv=True, request_rating=False, request_comments=False, user_name="user", experiment_name="default", path=""):
      """
      BaseLogCallback is a callback handler that logs the output of a chain.
      It can be used to log the output of a chain to a CSV file.
      The BaseLogCallback is used for anything that is common across chains and other langchain types, like agents.
      For example, saving files to a CSV is common across all types so it is defined here. 
      The params are as follows:
      output_csv: A boolean specifying whether to output to a csv.
      request_rating: A boolean specifying whether to request a rating from the user.
      request_comments: A boolean specifying whether to request comments from the user.
      user_name: A string specifying the user name.
      experiment_name: A string specifying the experiment name.
      path: A string specifying the path to where the CSV files should be saved.
      """
      print("Initialize the new instance of BaseLogCallback")
      super().__init__()
      # .log_history: Store all information about a log, including prompt template and model kwargs
      self.log_history = []
      # .responses only stores the input and output, plus the run ID so that 
      # users can use this to find the exact inputs used to generate this output 
      self.responses = [] 
      self.log_count = 0
      self.input_dict = {}
      self.output_csv = output_csv
      self.rating = None
      self.request_rating = request_rating
      self.request_comments = request_comments
      self.user_name = user_name
      self.experiment_name = experiment_name
      self.start_time = datetime.now()
      self.existing_data = {}
      self.path = path
    
    def create_responses_only_dict(self, info):
      """
      Because this callback saves a lot of information, a subset of fields are saved to a different csv.
      This creates that subset of fields. This can be used to quickly read through inputs and outputs to find the run_id that you like.
      """
      responses_only = {
         "run_id": info['run_id'],
         "input": info['input'],
         "response": info['response'],
         "start_time": info['start_time'],
         "duration_in_seconds": info['duration_in_seconds'],
      }
      if "rating" in info:
          responses_only["rating"] = info["rating"]
      if "comments" in info:
          responses_only["comments"] = info["comments"]
      return responses_only
    
    def add_to_log_history(self, info, create_responses_only_dict=True):
      """
      This function adds the constructed dictionary (info) to the log history.
      It also saves the dictionary to a CSV file if the output_csv is True.
      The create_responses_only_dict parameter is used to create a subset of the dictionary that only contains the fields that are needed for the responses_only csv.
      The responses_only csv is used to quickly read through inputs and outputs to find the run_id that you like.
      """

      if "children_ids" in info.keys():
        info.pop("children_ids", None)
      if self.output_csv:
          self.add_to_csv_history(info, "responses_all_fields")
          if create_responses_only_dict:
            responses_only = self.create_responses_only_dict(info)
            self.add_to_csv_history(responses_only, "responses_subset_of_fields")
            self.responses.append(responses_only)
      self.log_history.append(info)
      
    
    def add_to_csv_history(self, data, whats_logged):
        """
        add data to csv
        """ 
        file_name = self.path + self.user_name.replace(" ","_") + f"_{whats_logged}_{self.experiment_name}.csv"
        
        new_data = pd.json_normalize(data)

        handle_csv_column_diffs(file_name, new_data, whats_logged)

    
    def get_user_inputs(self, response):
      """
      This gets the inputs from the user for rating and comments. It returns {} if neither are desired.
      """
      user_inputs = {}
      if not self.request_rating and not self.request_comments:
        #  print("No user input required.")
         return user_inputs
      print("Response from llm:")
      print(response)
      
      # Adding a time.sleep to ensure the response has been printed before asking for user input.
      time.sleep(0.1)
      if self.request_rating:
        # sleep for 0.1 second to ensure the response has been printed
        user_inputs["rating"] = input("Rate the response?")
      if self.request_comments:
        user_inputs['comments'] = input("Please provide any comments on why you gave it that rating: ")
      return user_inputs
    
    def get_user_profile(self):
        """
        This can be used in sagemaker notebooks to get the user profile. 
        """
        try:
          # this only works in sagemaker studio
          with open("/opt/ml/metadata/resource-metadata.json", "r") as f:
              app_metadata = json.loads(f.read())
              # print(json.dumps(app_metadata, indent=2))
              sm_user_profile_name = app_metadata["UserProfileName"]
              return sm_user_profile_name
        except:
          return "unknown_user"

    # Keep the llm error pattern on the main logger so that if the LLM errors at any point during
    # other loggers, we track errors the same way. 
    def on_llm_error(self, error, **kwargs):
      """
      This handles if the LLM errors out. This is on the base logger but the callback still needs to be applied when creating the LLM in order to have it get invoked.
      """
      print(error)
      history_dict = {
          "type": "chain",
          **self.input_dict,
          "response": error,
          "error": "True"
      }
    
      self.add_to_log_history(history_dict, create_responses_only_dict=True)



def handle_csv_column_diffs(file_name, new_data, whats_logged):
  """
  This function handles the case where the new data has new columns that are not in the csv.
  It adds the new columns to the csv and saves it.
  It also handles the case where the new data has missing columns that are in the csv.
  It adds the missing columns to the csv and saves it.
  """
  #  check to see if we have new columns that dont exist in the csv that do exist in the new data
  use_header = not os.path.isfile(file_name)
  whats_logged_str = whats_logged.replace("_", " ")
  
  if not use_header:
    existing_columns = set(pd.read_csv(file_name, nrows=0).columns.tolist())
    columns_in_new_data = set(list(new_data.keys()))
    existing_data = pd.read_csv(file_name)
    # first, handle if the new data is missing any existing headers
    missing_headers = set(existing_columns) - set(columns_in_new_data)
    if len(missing_headers) > 0:
      # print(f"[info] newest data is missing the following headers: {missing_headers}")
      for header in missing_headers:
        new_data[header] = None
    
    # Now check if the new data has new headers
    new_headers = set(columns_in_new_data) - set(existing_columns)
    if len(new_headers) > 0:
      # print(f"[info] newest data has headers that do not exist in the csv: {new_headers}.\n[info] Adding the header to the csv...")
      #  there are new columns, so we need to add them to the existing data
      for header in new_headers:
        existing_data[header] = None
      
    # add new_data to existing_data
    full_df = pd.concat([existing_data, new_data])
    full_df.to_csv(file_name, mode='w', index=False, header=True)
    print(f"saved {whats_logged_str} to {file_name}")
    return
  # This code is only run if there are no new headers (only missing headers) or no existing data.
  new_data.to_csv(file_name, mode='a', index=False, header=use_header)
  
  print(f"saved {whats_logged_str} to {file_name}")
  return new_data
   
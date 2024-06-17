# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import re

import pandas as pd
import ast

from langchain.callbacks.base import BaseCallbackHandler
from datetime import datetime
from .prompt_engineering_logger import BaseLogCallback, handle_csv_column_diffs

class ChainLogCallback(BaseLogCallback):
    def __init__(self, output_csv=True, request_rating=False, request_comments=False, 
                 user_name="user", experiment_name="default", path="", input_keyword="question", 
                 combine_all_actions_into_one_log=True):
      """
      ChainLogCallback is a callback handler that logs the output of a chain.
      It can be used to log the output of a chain to a CSV file.
      The params are as follows:
      output_csv: A boolean that decides if you should output to csv
      request_rating: A boolean specifying whether to request a rating from the user.
      request_comments: A boolean specifying whether to request comments from the user.
      user_name: A string specifying the user name.
      experiment_name: A string specifying the experiment name.
      path: A string specifying the path to where the CSV files should be saved.
      input_keyword: A string specifying the keyword to use for the input.
      combine_all_actions_into_one_log: A boolean specifying whether to combine all actions into one log.
      The default is True.
      The log is saved to a CSV file with the following columns:
      - timestamp: The timestamp of the log.
      - inputs (multiple columns): The inputs to the chain. These vary base on chain type, and include model_id, prompt template, temperature, etc.
      - output: The output of the chain.
      - rating: The rating of the user.
      - comments: The comments of the user.
      """
      print("Initialize the new instance of ChainLogCallback")
      self.log_history = []
      self.responses = []
      self.input_dict = {}
      self.input_dict_additional_info = {}
      self.log_count = 0
      self.output_csv = output_csv
      self.rating = None
      self.request_rating = request_rating
      self.request_comments = request_comments
      self.user_name = user_name
      self.experiment_name = experiment_name
      self.existing_data = {}
      self.path = path
      self.input_keyword = input_keyword
      self.chain_type = ""
      self.primary_chain_id = ""
      self.combine_all_actions_into_one_log = combine_all_actions_into_one_log # this allows user to say if they want to log the intermediate chains
      self.chain_type_response_keys = {
          'ConversationalRetrievalChain': "answer",
          "LLMChain": "text",
          "StuffDocumentsChain": "output_text",
          "ConversationChain": "response"
      }
    
    def extract_repr(self, repr):
        """
        This function extracts the kwargs from the repr, which is the printable representational string of the given object, in this case the chain.
        It returns a dictionary of kwargs, which includes the temperature, the prompt template, etc. 
        The kwargs are extracted from the repr using a regular expression.
        """
        # print(f"*** repr: {repr}")
        str_with_kwargs = repr.split("(")[-1].split(")")[0].split(">, ")[1]
        regex = r"(?P<key>\w+)=(?P<value>(?:[^,{}]|\{[^{}]*\}|\btrue\b|\bfalse\b|\[.+\]|\w+)+)"
        matches = re.finditer(regex, str_with_kwargs, flags=re.DOTALL)
        chain_model_kwargs = {}
        for match in matches:
            key = match.group("key")
            value = match.group("value")
            chain_model_kwargs[key] = value
        verbose = chain_model_kwargs.pop('verbose', None)
        chain_model_kwargs = {"verbose": verbose, **chain_model_kwargs}
        return chain_model_kwargs
    
    def construct_input_dict_common(self, serialized, inputs, start_time: datetime, **kwargs):
        """
        This function constructs the input dictionary for the common aspects of various chains. 
        The chains have a lot of overlap in terms of structure, so this is used to create the common inputs from all tested chains.
        It takes in serialized, inputs, start_time, and kwargs.
        It doesn't return anything but it sets the input dictonary for the current run ID on the instantiated class object.
        """

        if not kwargs['parent_run_id']:
            self.primary_chain_id = self.current_run_id
        self.input_dict[self.current_run_id] = {}
        self.input_dict[self.current_run_id]['start_time'] = start_time
        self.input_dict[self.current_run_id]['run_id'] = str(kwargs['run_id'])
        
        # If we want to keep all logs separate, we need to tie the child logs to the parent log
        if not self.combine_all_actions_into_one_log:
            self.input_dict[self.current_run_id]['top_level_chain_run_id'] = self.primary_chain_id
            self.input_dict[self.current_run_id]['is_top_level_chain'] = self.primary_chain_id == self.current_run_id

        
        self.input_dict[self.current_run_id]['template_variables'] = list(inputs.keys())
        for variable in self.input_dict[self.current_run_id]['template_variables']:
            self.input_dict[self.current_run_id][variable + "_template_variable_value"] = inputs[variable]
        self.input_dict[self.current_run_id]['input'] = inputs[self.input_keyword]
        self.input_dict[self.current_run_id]['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.input_dict[self.current_run_id]['chain_type'] = serialized["id"][-1]
        self.input_dict[self.current_run_id]['chain_type_long'] = ".".join(serialized["id"])
        self.input_dict[self.current_run_id]['children_ids'] = []
        self.input_dict[self.current_run_id]['child_chains'] = []

        if kwargs['parent_run_id']:
            self.input_dict[self.primary_chain_id]['children_ids'].append(self.current_run_id)
            self.input_dict[self.primary_chain_id]['child_chains'].append(self.chain_type)
        self.input_dict_additional_info[self.current_run_id] = {}
        self.input_dict_additional_info[self.current_run_id]['chain_langchain_type'] = serialized["type"]



    def construct_input_dict_ConversationChain(self, serialized, inputs, **kwargs):
        """
        This extracts the prompt template and other inputs from the ConversationChain. 
        The way this is done for the ConversationChain is different than for other chains which is why the functions are separate.
        """
        self.input_dict[self.current_run_id]['prompt_template'] = serialized["kwargs"]["prompt"]['kwargs']['template']
        if "kwargs" in serialized['kwargs']['llm']:
            model_kwargs = serialized['kwargs']['llm']['kwargs']
            chain_model_kwargs = {}
        else:
            chain_model_kwargs = self.extract_repr(serialized['kwargs']['llm']['repr'])
            model_kwargs = ast.literal_eval(chain_model_kwargs['model_kwargs'])


        self.input_dict_additional_info[self.current_run_id]['prompt_type'] = ".".join(serialized["kwargs"]["prompt"]['id'])
        self.input_dict_additional_info[self.current_run_id]['prompt_langchain_type'] = serialized["kwargs"]["prompt"]['type']
        
        # self.input_dict_additional_info[self.current_run_id]['memory_type'] = ".".join(serialized["kwargs"]["memory"]['id'])
        # self.input_dict_additional_info[self.current_run_id]['memory_langchain_type'] = serialized["kwargs"]["memory"]['type']
        self.input_dict_additional_info[self.current_run_id]['llm_type'] = ".".join(serialized["kwargs"]["llm"]['id'])
        self.input_dict_additional_info[self.current_run_id]['llm_langchain_type'] = serialized["kwargs"]["llm"]['type']
        

        self.input_dict[self.current_run_id] = {**self.input_dict[self.current_run_id], **chain_model_kwargs, **model_kwargs,
                    "error": "False"}

    def get_match(self, serialized_str, pattern, group1=False):
        match = re.search(pattern, serialized_str)
        if not match:
            return None
        if group1:
            return match.group(1)
        return match.group()

    def get_retriever_info(self, serialized):
        retriever_info = {}
        serialized_str = serialized['repr']
        # print(serialized)
        retriever_str_long = self.get_match(serialized_str, r"retriever=\w+\(")
        retriever_full = self.get_match(serialized_str, r'retriever=(.+?)\)')
        retriever = retriever_str_long.replace("retriever=", "")[:-1]
        retriever_info['retriever'] = retriever
        if "AmazonKendraRetriever" in retriever_str_long:
            kendra_index_id = self.get_match(retriever_full, r"index_id='(.+?)'", group1=True)
            # print(kendra_index_id)
            retriever_info['kendra_index_id'] = kendra_index_id
        if "VectorStoreRetriever" in retriever_str_long:
            tags = self.get_match(serialized_str, r"tags=\[(.+?)\]", group1=True)
            tag_list = eval(tags)
            # print(tags)
            retriever_info['retriever_db_and_embeddings'] = tags
            retriever_info['retriever_db'] = tag_list[0]
            retriever_info['retriever_embeddings'] = tag_list[1]
        return retriever_info

    def construct_input_dict_ConversationalRetrievalChain(self, serialized, inputs, **kwargs):
        retriever_info = self.get_retriever_info(serialized)
        if retriever_info:
            self.input_dict_additional_info[self.current_run_id]['retriever'] = retriever_info['retriever']
            self.input_dict[self.current_run_id] = {**self.input_dict[self.current_run_id], 
                                                   **retriever_info,
                                                   }

        # TODO update this to find prompt template from CRC if not combine_all_actions_into_one_log
        return

    def construct_input_dict(self, serialized, inputs, start_time, **kwargs):
        """
        This function constructs the input dictionary for the given chain by calling the right construct input dict function based on the chain type.
        It takes in serialized, inputs, start_time, and kwargs.
        It doesn't return anything but it sets the input dictonary for the current run ID on the instantiated class object.
        """
        self.chain_type = serialized["id"][-1]
        self.construct_input_dict_common(serialized, inputs, start_time, **kwargs)
        if self.chain_type in ["ConversationChain", "LLMChain"]:
            self.construct_input_dict_ConversationChain(serialized, inputs, **kwargs)
        if self.chain_type == "ConversationalRetrievalChain":
            self.construct_input_dict_ConversationalRetrievalChain(serialized, inputs, **kwargs)

    def on_chain_start(self, serialized, inputs, **kwargs):
        """
        What to run when the chain starts. This is an implementation on top of the on_chain_start on the base handler.
        """
        start_time = datetime.now()
        self.current_run_id = str(kwargs['run_id'])
        self.log_count += 1 # do it at the start in case the chain fails
        self.construct_input_dict(serialized, inputs, start_time, **kwargs)
        # print(f"chain start @ {start_time}")
        # print(f"serialized: {serialized}")
        # prompt = repr(serialized['repr'])
        # print(f"inputs: {inputs}")
        # print(f"kwargs: {kwargs}")

    def on_chain_end(self, response, **kwargs):
        """
        What to run when the chain ends. This is an implementation on top of the on_chain_end on the base handler.
        This finds the duration, and saves the input and output to the desired format.
        """
        # TODO add token info if possible
        end_time = datetime.now()
        
        # Get the current run id and some other key information
        self.current_run_id = str(kwargs['run_id'])
        self.chain_type = self.input_dict[self.current_run_id]['chain_type']
        response_key = self.chain_type_response_keys[self.chain_type]

        # Now get the start time
        start_time_str = self.input_dict[self.current_run_id]['start_time']
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
        duration = (end_time - start_time).total_seconds()

        # Define use inputs
        user_inputs = {}
        if self.current_run_id == self.primary_chain_id:
            user_inputs = self.get_user_inputs(response[response_key])
        
        chain_history_dict = {
            "type": "chain",
            "response": response[response_key],
            **user_inputs,
            "duration_in_seconds": str(duration),
            **self.input_dict[self.current_run_id],
        }
        # If we want to keep all logs separate, we need to tie the child logs to the parent log
        # Currently this is only supported for ConversationalRetrievalChain
        is_parent_run = self.current_run_id == self.primary_chain_id
        if self.combine_all_actions_into_one_log and self.input_dict[self.primary_chain_id]['chain_type'] == "ConversationalRetrievalChain":
            # Only need to do anything if the current run is the primary chain (i.e. the first chain)
            if is_parent_run:
                chain_history_dict = self.add_children_to_primary_chain_dict(chain_history_dict)
                self.add_to_log_history(chain_history_dict, create_responses_only_dict=True)
        else:
            chain_history_dict = {**chain_history_dict, **self.input_dict_additional_info[self.current_run_id]}
            self.add_to_log_history(chain_history_dict, create_responses_only_dict=is_parent_run)
        self.log_count -= 1 # do it at the end because we want to keep the chain number from start for the last chain
    
    def add_children_to_primary_chain_dict(self, chain_history_dict):
        """
        This is how all of the child chains are combined into one entry for a conversationalRetrievalChain
        """
        for child in self.input_dict[self.primary_chain_id]['children_ids']:
            chain_history_dict = {**self.input_dict[child], **chain_history_dict}
        return chain_history_dict

    def on_chain_error(self, error, **kwargs):
        """
        What to run on chain error. This is an implementation on top of the on_chain_error in the base handler
        Note that it still saves information even if the chain encounters an error.
        """
        # TODO come back to this to determine how to handle errors when not saving all chains
        end_time = datetime.now()
        # Now get the start time
        self.current_run_id = str(kwargs['run_id'])
        start_time_str = self.input_dict[self.current_run_id]['start_time']
            
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
        duration = (end_time - start_time).total_seconds()
        print("chain error!!")
        print(error)
        history_dict = {
            "type": "chain",
            "primary_chain_id": self.primary_chain_id,
            **self.input_dict[self.current_run_id],
            "response": error,
            "duration_in_seconds": str(duration),
            "error": "True"
        }
        if self.combine_all_actions_into_one_log:
            history_dict = self.add_children_to_primary_chain_dict(history_dict)
        
        self.add_to_log_history(history_dict, create_responses_only_dict=True)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata= None, **kwargs):
        """
        For now, this is just used to show when the LLM starts if desired. 
        """
        print("llm starting")
    
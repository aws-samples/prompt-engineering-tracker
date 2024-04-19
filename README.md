# Prompt engineering tracker
Welcome to the prompt engineering tracker readme!

## Description
This was created to make it easier to do prompt engineering for ML engineers and data scientists, or anyone who is doing active prompt / model development in code (e.g. Jupyter notebooks). The `src/` folder contains files that create a custom callback that takes the inputs to a LangChain Chain as well as the outputs and writes them to an output location, such as a csv file. This has the following benefits:
- You don't need to record the prompt templates you run as you go. So, if you change one word in your prompt template, it will record it for you
- You can audit how many tests you have done against a chain
- You can share your results with teammates and stakeholders
- You can revisit previous runs to find inputs that yield good responses
- You can evaluate how different models perform

This is especially useful if you are trying out different chains or inputs to various chains and want to see how different LLMs / prompts / kwargs impact the response.


## Getting started

Check out the sample notebooks (`sample_ConversationalRetrievalChain_with_callback.ipynb` and `sample_ConversationChain_with_callback.ipynb`) to see examples of how to use the custom callback in your own code. 


## Installation
`pip install -r requirements.txt` will install the required packages. It is recommended to run the desired notebook to experiment with the callbacks. You can modify the inputs to the callback on instantiation. 

## Usage
To use the callback, simply instantiate it (example provided in the instantiation of `ChainLogCallback` inside the notebooks) and then pass the callback into the callbacks of whatever chain you create. An example of how to do this can be seen in the sample ConversationalRetrievalChain with callback notebook, specifically in the cells where the variables `callback` and `qa` are defined. 

To specify how you want to use the ChainLogCallback, you can change the input parameters:
* output_csv: A boolean that decides if you should output to csv. Allowed values: `True` or `False`
* request_rating: A boolean specifying whether to request a rating from the user. Allowed values: `True` or `False`
* request_comments: A boolean specifying whether to request comments from the user. Allowed values: `True` or `False`
* user_name: A string specifying the user name.
* experiment_name: A string specifying the experiment name.
* path: A string specifying the path to where the CSV files should be saved.
* input_keyword: A string specifying the keyword to use for the input.
* combine_all_actions_into_one_log: A boolean specifying whether to combine all actions into one log.
  The default is True.
  The log is saved to a CSV file with the following columns:
  - timestamp: The timestamp of the log.
  - inputs (multiple columns): The inputs to the chain. These vary base on chain type, and include model_id, prompt template, temperature, etc.
  - output: The output of the chain.
  - rating: The rating of the user.
  - comments: The comments of the user.

## Support
Please raise an issue for support. 

## Roadmap
Future actions include:
* adding tokens to the output, where possible
* outputting the responses to JSON
* outputting the responses to databases (e.g. Amazon DynamoDB)
* creating tool log callbacks
* creating agent log callbacks
* validating for additional chain types
* validating for other LLM providers


## License
Please see license file. 
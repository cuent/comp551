# Are your post popular?
#### COMP 551 [project 1](https://cs.mcgill.ca/~wlh/comp551/files/miniproject1_spec.pdf)

In this work we explore the performance of the linear regression model for  predicting  the  comment  popularity  on  the  social  networking  website  Reddit.   Higher the  popularity  of  the  comment  more  prominently  it  is  featured.   The  target  of  the  mini-project was to find the relation between certain features such as controversiality, whether the comment has replies to it, or a number of text features such as the most common words, the sentiment of the comment to the popularity of the comment.

# Install

Install dependencies with conda

`conda create -n newenvironment --file requirements.txt`

# Project structure

The project is structured in three folders. 
1. **data:** contains raw data (_reddit.json_), [datasets](data/process) built for models, and dictionary of words (_dict.csv, stopwords.txt, vader_lexicon.txt, words.txt_).
2. **result:** contains a csv file with results of all the models executed (*all_data.csv*) and information of the three required models (*main_data.csv*)
3. **src:** contains the source code of the project. The source code contain the following files.
    1. *tfidf.py*: Contains the implementation of TF-IDF.
    2. *metric.py*: Implementation of metrics, in this case MSE.
    2. *data_process.py*: Pipeline to process the dataset.
    3. *data_load.py (Main file)*: generates different [datasets](data/process) and stores them for future use.
    2. *learning_rate.py*: Implementation of learning rates (i.e. constant, decay, momentum)
    1. *linear_regression.py*: Implementation of linear regression using the closed-form and gradient descent approach.
    3. *model.py*: Executes a model given a set of parameters.
    4. *comp551_notebook.py*: analyses of the data and execution of models.

# Running

1. Generate datasets:

`python src/data_load.py`

2. Execute models:

`jupyter notebook .`
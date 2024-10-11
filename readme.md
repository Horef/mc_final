# Short Overview of the files in the repository
Note: any file that is not mentioned here is auxiliary and not crucial for the project.
- [readme.md](readme.md): this file
- [requirements.txt](requirements.txt): the file containing the required libraries for the project
- [main.py](main.py): the main python file where we train and test different models.
- [data_preprocessing.py](data_preprocessing.py): contains a few useful functions we have used to preprocess the data. 
Also splitting the data into folds and train and test sets
- [data_scraping.py](data_scraping.py): contains the functions we have used to scrape the data from the Reddit website.
- [data_exploration.ipynb](data_exploration.ipynb): a Jupyter notebook where we explore the data we have collected, 
and do the hierarchical analysis.
- [comments.csv](comments.csv): the file containing all the data of all the comments we have collected.
- [random_comments.csv](random_comments.csv): the file containing the data of the 50 randomly selected comments.
- [labels](labels): the folder containing the files with labels in two formats - csv and pkl. 
(pkl can be read by the pandas library.)
  - [first_25.csv](labels/first_25.csv): averaged labels for the first 25 comments
  - [second_25.csv](labels/second_25.csv): averaged labels for the second 25 comments
  - [full_data.csv](labels/full_data.csv): labels for all 50 comments
  - [ffn_data.csv](labels/ffn_data.csv): labels for the comments created by the FFN model
- [Latex Doc](Latex%20Doc): folder with the latex document for the project
- [models](models): folder containing our implementation of the different models
  - [Baseline](models/Baseline.py): folder containing the baseline model
  - [DecisionTree](models/DecisionTree.py): folder containing the Decision Tree model
  - [FeedForwardNetwork](models/FeedForwardNetwork.py): folder containing the Feed Forward Network model
  - [KNearestNeighbors](models/KNearestNeighbors.py): folder containing the K Nearest Neighbors model
  - [StatisticalMethods](models/StatisticalMethods.py): folder containing the methods for the statistical analysis 
  (in our case there is only the function for cross-validation)
- [plots](plots): folder containing all of the plots created in the process of the project.
  - The names of the plots are self-explanatory
  - [fnn_data](plots/fnn_data): folder containing the plots for the BEVoCI analysis of the FFN model
  - [full_data](plots/full_data): folder containing the plots for the BEVoCI analysis of the full data
- [raw_labels](raw_labels): folder containing the raw labels for the comments
  - [first_25](raw_labels/first_25): folder containing the raw labels for the first 25 comments (one file for each person)
  - [second_25](raw_labels/second_25): folder containing the raw labels for the second 25 comments (one file for each person)
- [To give in](To%20give%20in): folder with the final version of the project pdf
  - [report_gal_sergiy.pdf](To%20give%20in/report_gal_sergiy.pdf): the final version of the project pdf
  - [data_exploration.pdf](To%20give%20in/data_exploration.pdf): the pdf version of the data exploration notebook
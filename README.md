## Simple Modeling Experiments

This is my scratch repo that I am using to better understand the Data Science process, pieces involved and
hopefully increase my overall fluency in this exciting field.  

## Setting up Development

If you are going to clone from scratch you will need to clone with submodules using the below command   

        git clone --recurse-submodules -j8 https://github.com/mikekwright/simple-modeling-experiment.git 

If you are just grabbing the latest, after pulling be sure to get the latest submodules using the command   

        git submodule update --recursive --remote


## Sample Runs

### KNN

### LR

- Run using iris data and MSE
`python -m modeling --output /tmp/test2 train configs/kfold_template.json configs/models/lr/sv_lr.json configs/datasets/iris/petal_l_to_w.json configs/evaluators/mse.json`

## Study Material

There are a few books that I am using to further my understanding, they are:

* [Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)  
*

### Data Sources

There are a few different places to obtain workable datasets, I have listed a few of those that I am aware
here.  

* [UCI Repository](https://archive.ics.uci.edu/ml/datasets.html)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [Recommended by Category (2017)](https://elitedatascience.com/datasets)
* [Wikipedia Lists](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)   
* [Reddit Datsets](https://www.reddit.com/r/datasets)
* [Google Trends DataSources](http://googletrends.github.io/data/)  
* [USGS Gov DataSets](https://www.usgs.gov/products/data-and-tools/overview)  
* [Python Package for Datasets - Retriever](https://github.com/weecology/retriever) 

* [EliteDataScience - Learn Statistics for Data Science](https://elitedatascience.com/learn-statistics-for-data-science)

There is also a stackoverflow for datasets

* [OpenData](https://opendata.stackexchange.com/)  

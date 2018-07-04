## Simple Modeling Experiments

This is my scratch repo that I am using to better understand the Data Science process, pieces involved and
hopefully increase my overall fluency in this exciting field.  

## Setting up Development

If you are going to clone from scratch you will need to clone with submodules using the below command   

        git clone --recurse-submodules -j8 https://github.com/mikekwright/simple-modeling-experiment.git 

If you are just grabbing the latest, after pulling be sure to get the latest submodules using the command   

        git submodule update --recursive --remote
        
### Installing in WSL (Windows Subsystem Linux)

Since WSL is available directly on windows now, I am going to demonstrate how to install using `pipenv` and
the `ubuntu` distro.  

1. Install WSL (follow instructions from Microsoft)
2. Install the following dependencies using apt

    sudo apt install python3 python3-dev python3-venv python3-pip build-essential cmake pkg-config \
        libjpeg-dev libtiff-dev libpng-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libatlas-base-dev gfortran \
        libsm6 libxext6
        
3. Install `pipenv` globally using `pip3`

    sudo pip3 install pipenv
    
3b (optional). If pipenv is not found in your path you may need to add the following to your bash prompt

    PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"
    export PATH=$PATH:$PYTHON_BIN_PATH

4. Install virtualenv using `pipenv`

    pipenv install --dev
    
5. Start virtualenv using `pipenv shell`

#### Docker with windows and WSL

To get the docker environment correctly setup for WSL you will need to do the following. 

1. Enable hyper-v and containers in windows addon
2. Install Docker for windows
3. Enable the option in Docker for Windows for access from localhost without TSL
4. Add a DOCKER_HOST export to `~/.bashrc`

    export DOCKER_HOST=tcp://localhost:2375
    
5. Adjust the WSL to mount to `/` instead of `/mnt` so that the `C:\` drive is now `/c/`
    This allows for `docker run -v $PWD:/workdir` to function correctly

### Installing Scala Notebook (Windows)

**NOTE:** As of June 15, 2018 this solution only works for java 8 (9 and 10 currently fail).  

Install a version of java such as openjdk [OpenJDK](https://github.com/ojdkbuild/ojdkbuild)   

Download the scala-notebook install for windows from [here](https://github.com/rvilla87/Big-Data/raw/master/other/jupyter-Scala_2.11.11_kernel_Windows.zip), this
link is from the [following issue](https://github.com/jupyter-scala/jupyter-scala/issues/1)   

## Development practices

### Coding Styles

* [Python styler - Black](https://github.com/ambv/black) 


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


### Setup References

* [Install opencv in ubuntu](https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)
* [pipenv](https://docs.pipenv.org/basics/#example-pipfile-p)
* [WSL for VSCode Terminal](https://stackoverflow.com/questions/44450218/how-do-i-use-bash-on-ubuntu-on-windows-wsl-for-my-vs-code-terminal)
* [MobaXTerm - X11 for windows](https://mobaxterm.mobatek.net/)
* [WSL - Moba for Windows Linux Dev](https://nickjanetakis.com/blog/using-wsl-and-mobaxterm-to-create-a-linux-dev-environment-on-windows)
* [Setting up docker for WSL](https://nickjanetakis.com/blog/setting-up-docker-for-windows-and-wsl-to-work-flawlessly)
* [Mininet - Virtual Network on your laptop](http://mininet.org/)
* [WSL Startup Task for cmder](https://i1.wp.com/gingter.org/wp-content/uploads/2016/11/Bash-in-Cmder.png?ssl=1)
* [Pandoc - Convert documents between formats](https://pandoc.org/installing.html)


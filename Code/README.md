
# Code Instruction

## Original Data
**"DATA.csv"** and **"News.csv"** are the file we will use in our project.
**DATA.csv** includes the stock price, economic index.
**News.csv** is the file we got from running the code in the **NLP** file (web_scraping and finbert_training)

## Load Data
After download the two original data, we can run the step 1 **"1. Load_data.py"**  
In this file, it will get two more features: Technical indicator and Fourier transform

## Data Preprocessing
In **"2. data_preprocessing.py"**, we will get the features and target for our project, and deal with NA value, then normalize our data.  
In this process, there are two parameters we can set here: **n_steps_in** and **n_steps_out**, we can set how many days we would like to inputm an how many day/days we would like to predict for the future.  

After we run this file, we will get X_train.npy, y_train.npy, X_test.npy, y_test.npy, yc_train.npy, yc_test.npy, index_train.npy and index_test.npy, that w may need to use in the following process


## Baseline LSTM
**"3. Baseline_LSTM.py"** is the baseline model for our project, which we use LSTM to d the prediction.

## Basic GRU
**"3. Basic_GRU.py"** is the traditional basic model for our project, which we use GRU to d the prediction.

## Basic GAN
For Basic GAN, **"4. Basic_GAN.py"** is our basic GAN model.

## WGAN-GP
**"5. WGAN_GP.py"** is the WGAN-GP model.

## Test prediction

Afte we train all th models, we can save the best model, and run **"6. Test_prediction.py "** to get th prediction.

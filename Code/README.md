
# Code Instruction

## Original Data
**DATA.csv** and **News.csv** are the file we will use in our project.
**DATA.csv** includes the stock price, economic index.
**News.csv** is the file we got from running the code in the **NLP** file (web_scraping and finbert_training)

## Load Data
After download the two original data, we can run the step 1 **1. Load_data.py**  
In this file, it will get two more features: Technical indicator and Fourier transform

## Data Preprocessing
In **"2. data_preprocessing.py"**, we will get the features and target for our project, and deal with NA value, then normalize our data.  
In this process, there are two parameters we can set here: **n_steps_in** and **n_steps_out**, we can set how many days we would like to inputm an how many day/days we would like to predict for the future.  

After we run this file, we will get X_train.npy, y_train.npy, X_test.npy, y_test.npy, yc_train.npy, yc_test.npy, index_train.npy and index_test.npy, that w may need to use in the following process


## WGAN
For running WGAN, please use the code **"WGAN.py"**

## WGAN-GP
For running WGAN-GP, please use the code **"WGANGP.py"**


After running each model, you will get a saved model called **"__GAN_g.pth"** and **"__GAN_d.pth"**
## Generate Images
After training the models, you can run the **"Generate_Images_.py"** to test how the model performs.  
In this code, you need to change the **"__GAN_g.pth"** file name at the bottom which included in the load pretrained model part.  
And also you can change the file name **"___.jpg"** at the bottom included in the save image part.


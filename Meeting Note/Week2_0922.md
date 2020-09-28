# Week2 - 0922 Meeting note:

## 1. What is the detail of the struction, what will be input in LSTM, what will be output?  
    a. Input data  
        Stock price data: Apple real stock price  
        Features: Stock Index  
                  Economic Index  
                  Technical Indicator  
                  Fourier Transform  
                  Daily News  
                  
    b. LSTM (Stacked LSTM) 
       Use Multivariate LSTM Model
       
       gan_num_features = dataset_total_df.shape[1]
        sequence_length = 17

        class RNNModel(gluon.Block):

         def __init__(self, num_embed, num_hidden, num_layers, bidirectional=False, 
                 sequence_length=sequence_length, **kwargs):
             super(RNNModel, self).__init__(**kwargs)
                self.num_hidden = num_hidden
                with self.name_scope():
                   self.rnn = rnn.LSTM(num_hidden, num_layers, input_size=num_embed, 
                                bidirectional=bidirectional, layout='TNC')     
            
                   self.decoder = nn.Dense(1, in_units=num_hidden)      ## One dimension output
    
         def forward(self, inputs, hidden):
              output, hidden = self.rnn(inputs, hidden)
              decoded = self.decoder(output.reshape((-1, self.num_hidden)))
             return decoded, hidden
    
         def begin_state(self, *args, **kwargs):
             return self.rnn.begin_state(*args, **kwargs)
    
         lstm_model = RNNModel(num_embed=gan_num_features, num_hidden=500, num_layers=1)
    
       
 ![image1](https://github.com/hungchun-lin/Stock-price-prediction-using-GAN-Capstone-Group1/blob/master/Meeting%20Note/images/Stacked_LSTM_Structure.png)

    c. GAN  
 ![image2](https://github.com/hungchun-lin/Stock-price-prediction-using-GAN-Capstone-Group1/blob/master/Meeting%20Note/images/GAN%20architecture.png)
## 2. Contribution?
    Baseline: Th Stanford report
    Contribution: Use the different GANs (WGAN and WGAN-GP) & Use more features 
                
               
## 3. Github: Folder: 
     We should creat some files: Paper/ Code/ Meeting note

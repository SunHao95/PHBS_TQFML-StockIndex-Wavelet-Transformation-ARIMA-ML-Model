# PHBS_TQFML-Project

## Stock Index Prediction Based on Wavelet Transformation and ARMA-ML Model or TEA-ML Model

* ### Wavelet Transformation
  Stock index data generally has much noise and is non-stationary, which is a huge challenge for us using ML(Machine Learning) methods to  predict the index. However wavelet transformation, an upgraded version of fourier transformation, can serve as a very good filter to decrease the noise in stock index and smooth the data, thus helping us to focus more on the main trend of stock index.
  
  <div align=center><img width="800" height="300" src="https://github.com/SunHao95/PHBS_TQFML-Stock-Index-Prediction-Based-on-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/1.png"/></div>
  <div align=center>Figure 1. Filter Bank Scheme for DWT</div>
  

  In the figure, H,L,and H’,L’ are the high-pass and low-pass filters for wavelet decomposition and reconstruction respectively. In the decomposition phase, the low-pass filter removes the higher frequency components of the signal and highpass filter picks up the remaining parts. Then, the filtered signals are downsampled by two and the results are called __approximation coefficients__ and __detail coefficients__. The reconstruction is just a reversed process of the decomposition and for perfect reconstruction filter banks, we have x = x'. A signal can be further decomposed by cascade algorithm as shown in following equation:

<div align=center><img width="300" height="200" src="https://github.com/SunHao95/PHBS_TQFML-Stock-Index-Prediction-Based-on-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/3.png"/></div>
 
  
  <div align=center><img width="400" height="350" src="https://github.com/SunHao95/PHBS_TQFML-Stock-Index-Prediction-Based-on-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/2.png"/></div>
  <div align=center>Figure 2. Wavelet Decomposition Tree</div>


 
  
* Prediction Model.
  * the ARMA-ML(Autoregressive Moving Average and Machine Learning) Model
  
  After wavelet transformation, there are two types of stock index data, low-frequency and high-frequency. The ARMA-ML model is trying to  using ARMA method to predict the high-frequency data,the __detail coefficients__, since high-frequency is stationary. While ML methods, such as SVR(Support Vector Regression) and GBR(Gradient Boosting Regression)，are trying to predict the low-frequency data, the __approximation coefficients__. Finally, using the predicted data together to reconstruct the stock index. Generally speaking, ARMA-ML model is trying to complete prediction on the timing series perspective.
 
 * ### ARMA Model 
 <a href="https://www.codecogs.com/eqnedit.php?latex=ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/png.latex?ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" title="ax^{2} + by^{2} + c = 0" /></a>



## Data Description
  The datas selected are the daily stock index data of 000300.SH representing the large-cap stocks and 000905.SH representing medium-and-small-cap stocks, including,
  * __Open__: Open daily price
  * __High__: Highest daily price
  * __Low__: Lowest daily price
  * __Close__: Close daily price
  * __Volume__: Trading volume
  * __AMT__: Trading amount
  * __Time range__: 2010-01-01 to 2018-03-30
  
  
  
 ## Arrangement
* Calculate around 10 technical indicators based on the __OHLC Price__, __Volume__ and __AMT__.
* Try best to use both models to make predictions. At least use one of them.
* Try to design a trading strategy based on the stock index prediction result.

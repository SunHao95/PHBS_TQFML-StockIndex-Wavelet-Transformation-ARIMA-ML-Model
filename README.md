# PHBS_TQFML-Project

# [StockIndex Prediction Based on Wavelet Transformation ARIMA-ML Model](https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/Project/Final%20Project.ipynb)

## Methodology
* ### Wavelet Transformation
  Stock index data generally has much noise and is non-stationary, which is a huge challenge for us using ML(Machine Learning) methods to  predict the index. However wavelet transformation, an upgraded version of fourier transformation, can serve as a very good filter to decrease the noise in stock index and smooth the data, thus helping us to focus more on the main trend of stock index.
  
  <div align=center><img width="800" height="300" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/1.png"/></div>
  <div align=center>Figure 1. Filter Bank Scheme for DWT</div>
  

  In the figure, H,L,and H’,L’ are the high-pass and low-pass filters for wavelet decomposition and reconstruction respectively. In the decomposition phase, the low-pass filter removes the higher frequency components of the signal and highpass filter picks up the remaining parts. Then, the filtered signals are downsampled by two and the results are called __approximation coefficients__ and __detail coefficients__. The reconstruction is just a reversed process of the decomposition and for perfect reconstruction filter banks, we have x = x'. A signal can be further decomposed by cascade algorithm as shown in following equation:

<div align=center><img width="500" height="150" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/3.png"/></div>
 
  
  <div align=center><img width="400" height="350" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/2.png"/></div>
  <div align=center>Figure 2. Wavelet Decomposition Tree</div>


 
  
* ### Prediction Model.
 * #### ARMA-ML(Autoregressive Moving Average and Machine Learning) Model
  
  After wavelet transformation, there are two types of stock index data, low-frequency and high-frequency. The ARMA-ML model is trying to  using ARMA method to predict the high-frequency data,the __detail coefficients__, since high-frequency is stationary. While ML methods, such as SVR(Support Vector Regression) and GBR(Gradient Boosting Regression)，are trying to predict the low-frequency data, the __approximation coefficients__. Finally, using the predicted data together to reconstruct the stock index. Generally speaking, ARMA-ML model is trying to complete prediction on the timing series perspective.
 
 * #### [ARMA](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model) 
 <div align=center><img src="https://latex.codecogs.com/png.latex?\bg_white&space;Z_{t}&space;=&space;\varphi_{1}Z_{t-1}&plus;\varphi_{2}Z_{t-2}&plus;\cdots&plus;\varphi_{p}Z_{t-p}&plus;a_{t}-\theta&space;_{1}a_{t-1}-\cdots-\theta&space;_{q}a_{t-q}" title="Z_{t} = \varphi_{1}Z_{t-1}+\varphi_{2}Z_{t-2}+\cdots+\varphi_{p}Z_{t-p}+a_{t}-\theta _{1}a_{t-1}-\cdots-\theta _{q}a_{t-q}" /></div>

Finding appropriate values of p and q in the ARMA(p,q) model can be facilitated by plotting the partial autocorrelation functions for an estimate of p, and likewise using the autocorrelation functions for an estimate of q. Further information can be gleaned by considering the same functions for the residuals of a model fitted with an initial selection of p and q.
Brockwell & Davis recommend using AICc for finding p and q

* #### [SVR](https://en.wikipedia.org/wiki/Support_vector_machine#Regression)
 Support vector regression (SVR) is a version of SVM for regression. The model produced by support vector classification (as described above) depends only on a subset of the training data, because the cost function for building the model does not care about training points that lie beyond the margin. Analogously, the model produced by SVR depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction.
 
 * #### [GBR](https://en.wikipedia.org/wiki/Gradient_boosting)
 Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.




## Data & Prediction
  The datas selected are the daily stock index data of 000300.SH representing the large-cap stocks and 000905.SH representing medium-and-small-cap stocks, including,
  * __Open__: Open daily price
  * __High__: Highest daily price
  * __Low__: Lowest daily price
  * __Close__: Close daily price
  * __Volume__: Trading volume
  * __AMT__: Trading amount
  * __Time range__: 2010-01-01 to 2018-03-30
  
  Use the former 4 days' close price to predict the next day's close price. Using 150-day rolling windown to make prediction. Finally, try to make a prediction of 30-day close price.
  
  ## Result
  * ### Wavelet Transformation
  <div align=center><img width="500" height="1000" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/4.png"/></div>
  <div align=center>Figure 3. Approximation&Detail Components of Wavelet Decomposition</div>
  
  * ### ARMA
  <div align=center><img width="500" height="150" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/5.png"/></div>
  <div align=center>Figure 4. ARMA Fit</div>
  
  * ### GBR Prediction
  <div align=center><img width="500" height="150" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/6.png"/></div>
  <div align=center>Figure 5. GBR Prediction</div>
  
  * ### SVR Prediction
  <div align=center><img width="500" height="150" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/7.png"/></div>
  <div align=center>Figure 6. SVR Prediction</div>
  
  
  * ### SVR_GBR Prediction
  <div align=center><img width="500" height="150" src="https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/master/images/8.png"/></div>
  <div align=center>Figure 7. SVR_GBR Prediction Prediction</div>
  
  * ### Evaluation
  Use common regression matrices to evaluate the results. 
   
    
##  Motivation & References
Stock index, as time series, inspires a lot of research to implement the forecast both in academic area and financial departments. Generally speaking, the main methods used to do prediction are time-series analysis and machine learning models. Some of the research reports and papers have presented good ideas to predict stock index by means of combined_models, such as TS &  ML models. Some even use some data processing methods like Wavelet Transformation to make the data properties more suitable to different predictin models. All the reference papers and research reports have been uploaded in the  [reference](https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/tree/master/reference) folder.

  

## Conclusion
 Unfortunately, it seems that none of the model has good prediction power, which indicates that stock prices cannot be predicted exactly!
 However, the "noisy" data processing methods and time-series analysis model as well as nonlinear machine learning regression model can
 serve as some useful tools to do further research in other fields.
* GBR prediction seems as the lag of previous stock prices, just predicting like a martingale.
* SVR performs badly in the begining of stock index prediction. As time goes by, it tends to predict the average(or expectation) price.
* The mix GBR/SVR model is just the simple mean of GBR and SVR. Its performance lies between GBR and SVR

  
  


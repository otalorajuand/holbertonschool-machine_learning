# Time Series Forecasting

> In this project, I use RNNs to estimate the close price of bitcoin (BTC). I followed the tutorial from the TensorFlow documentation and come up with interesting results. 

At the end of this project I was able to solve these conceptual questions:

* What is time series forecasting?
* What is a stationary process?
* What is a sliding window?
* How to preprocess time series data
* How to create a data pipeline in tensorflow for time series data
* How to perform time series forecasting with RNNs in tensorflow

## Tasks :heavy_check_mark:

| Filename | Task |
| ------ | ------------------------------------------------- | 
| [preprocess_data.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/preprocess_data.py)| This file preprocess the data, does feature engineering and splits the data into training and validation. |
| [window.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/window.py)| This file creates the class WindowGenerator which handles the creation of the dataset and the batches. |
| [forecast_btc.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/forecast_btc.py)| The main where the forecast is being done. This file imports the classes created in the preprocessing and window files. It builds and trains the model. |
| [BTC_forecasting.ipynb](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/BTC_forecasting.ipynb)| This is a notebook with plots and the whole process documented. |

### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd supervised_learning/time_series
```
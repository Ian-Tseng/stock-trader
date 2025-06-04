# Stock Trader
A Python-based trading strategy that leverages LSTM modeling to identify price stages and guide buy/sell decisions based on seasonal patterns.



## Prerequisite
- Python 3.10

### Installation
Clone the repository and navigate to the project directory.

Install dependencies:
```
pip install -r requirements.txt
```

### Running the Model
```
python trader.py --training training_data.csv --testing testing_data.csv
```



### Strategy Overview 
```
This strategy identifies and leverages seasonal price behavior patterns:

Period Segmentation:
Split the dataset into periods where local price deltas exceed the overall delta.

Stage Assignment:
Assign each price a stage from 0 to 7 within its period:

Stage 0: Lowest price in the period

Stage 7: Highest price in the period

Other stages are linearly interpolated between these extremes

Model Training:
Train an LSTM model to predict the stage of the next price point.

Decision Rules:

Buy if predicted stage < 3

Sell if predicted stage > 4

Hold otherwise


```


## AUTHORS
[Ian-Tseng](https://github.com/Ian-Tseng/)

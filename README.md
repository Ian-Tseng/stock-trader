# Stock Trader


## Prerequisite
- Python 3.10

### Install Dependency
```
requirements.txt
```

### Start Virtual Environment
```
pip install -r requirements.txt
python trader.py --training training_data.csv --testing testing_data.csv
```



### Idea 
```
Identify seasonal price deltas that are greater than the overall dataset’s price delta, and split the data into different life periods accordingly.

Assign a stage (0–7) to each price based on its position within the period.

Stage 0 = lowest price in that period

Stage 7 = highest price in that period

Stages are relative to the price range within each period.

Train an LSTM model to predict the price stage.

Generate actions based on the predicted stage:

Buy if stage < 3

Sell if stage > 4

```


## AUTHORS
[Ian-Tseng](https://github.com/Ian-Tseng/)

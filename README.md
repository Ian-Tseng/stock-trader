# Trader

DSAI 2022 HW1

## Prerequisite
- Python 3.10

### Install Dependency
```
requirements.txt
```

### Start Virtual Environment
```conda environment
```


### Idea 
```
1.	Find the price delta in a season if it greater than whole data’s price delta and split them into different life period.

2.	Tag the price a different stage every day and every period. It use from 0 to 7 in this data, 0 means the lowest price and 7 means the highest price ever appears in that period. In other words, the stage depends on the data in which period.

3. Train data with LSTM model to predict the stage.

4.	The action generate depends on stage level. If stage < 3 then will buy else if stage > 4 then will sell.

```


## AUTHORS
[Ian-Tseng](https://github.com/Ian-Tseng/)

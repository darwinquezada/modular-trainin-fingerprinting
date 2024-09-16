0. **Install requirements**
```sh
  pip install -r requirements.txt
```
1. **Datasets structure**
The structure of the datasets is:
* AP1, AP2, AP3, ..., APn, LONGITUDE, LATITUDE, ALTITUDE, FLOOR, BUILDINGID

2. **General parameters**
  * --config-file : Datasets and model's configuration (see config.json)
  * -- dataset : Dataset (e.g., UJI1)
  * -- algorithm : {CNN-LSTM|CNNLoc|KNN}

3. **Training model**
  * Run the following command:
```sh
  python main.py --config-file config.json --dataset UJI1 --algorithm CNN-LSTM
```
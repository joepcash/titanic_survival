import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from exploration import explore
from training import train
from prediction import predict

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# explore(train_data)

x = train_data.drop("Survived", axis=1)
y = train_data["Survived"]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)
train(x_train, x_validation, y_train, y_validation)
predict(x_train, x_validation, y_train, y_validation)
predict(x_train, test_data, y_train, None, True)
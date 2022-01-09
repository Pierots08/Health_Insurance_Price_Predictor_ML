# We load the neccesary libraries

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

# we load the clean data

df = pd.read_csv('data/processed/Insurance_treated_data.csv')


# Nombramos el target y las features

X = df.drop('charges', axis=1)
y = df['charges']

# Hacemos el split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Construimos el modelo

rf = RandomForestRegressor(bootstrap= True, max_depth= 110, max_features= 3, min_samples_leaf= 3, min_samples_split= 8, n_estimators= 200)

# Entrenamos los modelos

rf.fit(X_train, y_train)

# Evaluamos el modelo

predictions = rf.predict(X_test)
print('The model performance:' ) 
print('-------------------------------')
print('SCORE:', rf.score(X_test, y_test))
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))


# guardamos el modelo

model = rf
# Save Model Using Pickle
# We save the train Random Forest Regressor model as final_model_pkl
with open('model/final_model_pkl', 'wb') as files:
    pickle.dump(model, files)

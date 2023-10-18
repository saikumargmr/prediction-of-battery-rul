simport pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from scipy.stats import skew
uploaded = files.upload()
df = pd.read_csv('Battery_RUL.csv')
df.skew()
df=df.drop_duplicates()
df=df.dropna()
df.shape
df=df.drop('Cycle_Index',axis=1)
li=[]
for i in df.columns.values:
  if i!='RUL':
    li.append(i)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#splitting data set into train set and test set
from sklearn.model_selection import train_test_split
X = df.drop('RUL', axis=1)
y=df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test=X_test
testy=y_test
#standerdising
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
knn = KNeighborsRegressor(
    n_neighbors=3,
    weights='distance',
    p=1,
    metric='manhattan',
    algorithm= 'ball_tree',
    leaf_size= 20,
)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
            
            #model evalution
            #use this code for other algorthms to model evalution
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_squared_log_error, explained_variance_score,mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")
medae = median_absolute_error(y_test, y_pred)
print(f"Median Absolute Error (MedAE): {medae}")
msle = mean_squared_log_error(y_test, y_pred)
print(f"Mean Squared Logarithmic Error (MSLE): {msle}")
evs = explained_variance_score(y_test, y_pred)
print(f"Explained Variance Score: {evs}")
            
            #to compare prediction and original results
            #use this code for other algorthms to plot prediction and original results
for feature in li:
  plt.figure(figsize=(8, 6))
  plt.scatter(y_pred, X_test[feature], color='red',s=5,marker='x', label='y_pred')
  plt.scatter(y_test, X_test[feature], color='blue',s=1,marker='x', label='y_test')
  plt.xlabel('RUL')
  plt.ylabel(feature)
  plt.title('Scatter plot of {} vs. RUL'.format(feature))
  plt.legend()
  plt.grid(True)
  plt.show()
            
            #xg boost
            
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor = XGBRegressor(n_estimators=300, learning_rate=0.2, max_depth=5,min_child_weight=2)
xgb_regressor.fit(X_train, y_train)
y_pred = xgb_regressor.predict(X_test)
            
            #-decision tree
            
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
            
            #random forest
            
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
rf_regressor = RandomForestRegressor(
                                        n_estimators=100,
                                        max_depth=None,
                                        random_state=42
                                      )
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

            #naive bays
            
from sklearn.naive_bayes import GaussianNB
nb_regressor = GaussianNB()
nb_regressor.fit(X_train, y_train)
y_pred = nb_regressor.predict(X_test)


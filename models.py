from data import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklego.meta import ZeroInflatedRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = read_data(is_small=False) # 'tx_crash.csv' the new feature does not seem useful, acc=0.815

df_X = df.drop(columns=['tot_crash_count', 'crash'])
df_y = df['crash']

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

zir = ZeroInflatedRegressor(
    classifier=RandomForestClassifier(),
    regressor=RandomForestRegressor()
)

zir.fit(X_train, y_train)
y_val_pred = zir.predict(X_val)
plt.plot(y_val, y_val_pred, '.')
plt.show()

y_test_pred = zir.predict(X_test)
plt.scatter(y_test, y_test_pred)
plt.show()

zir.score(X_train, y_train)
zir.score(X_val, y_val)
zir.score(X_test, y_test)
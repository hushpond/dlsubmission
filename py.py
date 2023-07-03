#plot으로 corr 파악
plt.figure(figsize = (15, 10))
sns.heatmap(data = df_train.corr(), mask=np.triu(df_train.corr()), annot=True,
            cmap = 'RdBu_r', linewidth=2, linecolor='white')
plt.show()

#price와 강한 상관계수 보이는 변수들을 price와의 관계로 시각화
bathrooms = df_train['bathrooms'].values
sqft_living = df_train['sqft_living'].values
grade = df_train['grade'].values
sqft_above =  df_train['sqft_above'].values
sqft_living15 = df_train['sqft_living15'].values
plot_cols = [ 'bathrooms', 'sqft_living','grade', 'sqft_above', 'sqft_living15']
plot_df = df_train.loc[:, plot_cols]

train_price = df_train['price'].values
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
  ax1=plt.subplot(2, 2, idx+1)
  sns.regplot(x=col, y=train_price, data=plot_df, ax=ax1)
plt.show()


#Model 2. Lasso
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('lassomodel', Lasso(alpha=0.012, fit_intercept=True, max_iter=3000))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(x_train, y_train)

mse_scores_lasso = cross_val_score(lasso_pipe, x_val, y_val, scoring='neg_mean_squared_error')
rmse_scores_lasso = np.sqrt(-1 * mse_scores_lasso)
avg_rmse = np.mean(rmse_scores_lasso)
print(avg_rmse)

#Model 4. Keras

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(x=x_train,y=y_train,
          batch_size=128,epochs=400)
model.summary()

#Keras에서 epoch에 따른 학습 변화 확인 plot
loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

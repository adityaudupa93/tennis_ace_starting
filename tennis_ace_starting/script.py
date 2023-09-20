#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_data = pd.read_csv(r'C:\Users\adityau\Documents\GitHub\tennis_ace_starting\tennis_ace_starting\tennis_stats.csv')
# First 5 rows
print("First 5 rows:")
print(tennis_data.head())
# Info of the columns
print("-------------Information regarding the columns-----------")
print(tennis_data.info())
# Description of columns
print("---------Statistics of the columns-------------")
print(tennis_data.describe())

# Is there NaNs?
print("------------NaNs--------------")
print("Number of NaNs in each column")
print(tennis_data.isna().any())



# perform exploratory analysis here:

# Seperating columns
# 
# print(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith('First')) | (tennis_data.columns.str.startswith('Win'))]].head())


# Pairplots for all variables
relate_var = 'First'

# sns.pairplot(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith(relate_var)) | (tennis_data.columns.str.startswith('Win'))]])
# plt.show()
# plt.clf()

# sns.pairplot(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith(relate_var)) | (tennis_data.columns.str.startswith('Loss'))]])
# plt.show()
# plt.clf()

# sns.pairplot(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith(relate_var)) | (tennis_data.columns.str.startswith('Rank'))]])
# plt.show()
# plt.clf()



## perform single and two feature linear regressions here:
feat = ['ReturnGamesPlayed', 'BreakPointsOpportunities']
outc = ['Winnings']
features = tennis_data[feat]
outcome = tennis_data[outc]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train,outcome_train)

print("R2 score of predicting {} from {}: {}".format(outc, feat, model.score(features_test,outcome_test)))


prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title("Predicting {} from {}".format(outc, feat))
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


## perform multiple feature linear regressions here:


feat = ['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']
outc = ['Winnings']
features = tennis_data[feat]
outcome = tennis_data[outc]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train,outcome_train)

print("R2 score of predicting {} from {}: {}".format(outc, feat, model.score(features_test,outcome_test)))
print("Model parameters: {}".format(model.coef_))


prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title("Predicting {} from {}".format(outc, feat))
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
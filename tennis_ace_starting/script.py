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

# Seperating columns
# 
# print(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith('First')) | (tennis_data.columns.str.startswith('Win'))]].head())


# Pairplots for all variables
sns.pairplot(tennis_data.loc[:, tennis_data.columns[(tennis_data.columns.str.startswith('First')) | (tennis_data.columns.str.startswith('Win'))]], hue="Winnings")
plt.show()









# perform exploratory analysis here:






















## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:

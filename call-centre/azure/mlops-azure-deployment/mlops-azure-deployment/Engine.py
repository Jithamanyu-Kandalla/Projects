import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from MLPIpeline.MLP import MLP
from MLPIpeline.CNN_LSTM import CNN_LSTM
from MLPIpeline.CNN_Model import CNN_model
from MLPIpeline.LSTM import LSTM_model

warnings.filterwarnings("ignore")

# importing the data
raw_csv_data = pd.read_excel("Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()


# taken as a date time field

df_comp.set_index("month", inplace=True)



# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')



# ## Time Series Visualization

df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig("output/"+"healthcare.png")


df_comp.Telecom.plot(figsize=(20,5), title="Telecom")
plt.savefig("output/"+"telecom.png")


df_comp.Banking.plot(figsize=(20,5), title="Banking")
plt.savefig("output/"+"banking.png")


df_comp.Technology.plot(figsize=(20,5), title="Technology")
plt.savefig("output/"+"tech.png")


df_comp.Insurance.plot(figsize=(20,5), title="Insurance")
plt.savefig("output/"+"insurance.png")


# ## Setting the training format

data = df_comp.Healthcare


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 

#Empty lists to be populated using formatted training data
target_data = []

# Number of days we want to look into the future based on the past days.
n_past = 5  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (?)
#refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(len(data)):
    temp = []
    for j in range(n_past + 1):
        try:
            temp.append(data[i+j])
        except Exception as e:
            continue
    if len(temp) > 5:
        target_data.append(temp)

len(target_data)


# ## Train Test Split

import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import MinMaxScaler
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(target_data)
target_data_scaled = scaler.transform(target_data)


data_df = pd.DataFrame(target_data_scaled, columns=["t-4","t-3","t-2","t-1","t","Y"])


data_df.head()


# train set split
test_size = 20

train = data_df[:-test_size]
test = data_df[-test_size:]



X, Y = train.drop("Y", axis=1).values, train["Y"]
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=0)


X_test, Y_test = test.drop("Y", axis=1).values, test["Y"]

MLP(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)

CNN_model(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)

LSTM_model(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)

CNN_LSTM(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)




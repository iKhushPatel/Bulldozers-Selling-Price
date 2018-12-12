
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import datetime
from datetime import date
dataset = pd.read_csv("Train.csv")
#This is for Small data
df = dataset.head(20000)

#For whole prediction
df = dataset

#Description Column Dropped
drop_cols = ["fiProductClassDesc","fiModelDesc","fiBaseModel","fiSecondaryDesc","fiModelSeries","fiModelDescriptor","state"]

for i in drop_cols:
    df = df.drop(i,axis = 1)

#Drop Machine Hours Count = 0 
df = df.drop(df.index[df.MachineHoursCurrentMeter == 0.0])

cols = df.columns
cols = cols[4:]

d={}
for i in cols:
    item = df[i].value_counts().idxmax()
    d.update({i:item})
#print(d)

#All Missing Values are filed with most common values
for i in cols:   
    df[i] = df[i].replace(np.nan,d.get(i))

cols = ["datasource","auctioneerID","UsageBand","ProductSize","ProductGroup","ProductGroupDesc","Drive_System","Enclosure","Forks","Pad_Type","Ride_Control","Stick","Transmission","Turbocharged","Blade_Extension","Blade_Width","Enclosure_Type","Engine_Horsepower","Hydraulics","Pushblock","Ripper","Scarifier","Tip_Control","Tire_Size","Coupler","Coupler_System","Grouser_Tracks","Hydraulics_Flow","Track_Type","Undercarriage_Pad_Width","Stick_Length","Thumb","Pattern_Changer","Grouser_Type","Backhoe_Mounting","Blade_Type","Travel_Controls","Differential_Type","Steering_Controls"]

for i in cols:
    unique = df[i].unique()
    for j in unique:
        string = str(i)+"_"+str(j)
        df[string] = df[i].map(lambda x: 1 if x == j else 0)
    df = df.drop(i,axis=1)

#Getting days from SaleDate
total_days=[]
for i in df.saledate:
    x = datetime.datetime.strptime(i, '%m/%d/%Y %H:%M').date()
    l_date = date(1, 1, 1)
    delta = x - l_date
    total_days.append(delta.days)

df['SaleDate'] = total_days
fields = list(df.columns)
df.to_csv('NewDataset.csv',index=False)


fields.remove('SalesID')
fields.remove('MachineID')
fields.remove('ModelID')
df = pd.read_csv("NewDataset.csv",usecols=fields)

y = df.SalePrice

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.55)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
pred1=model.predict(x_test)
print("Score is :: ", model.score(x_test, y_test))


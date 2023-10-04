import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

match=pd.read_csv('Match.csv')
delivery=pd.read_csv('Delivery.csv')

total_score= delivery.groupby(['ID','innings']).sum()['total_run'].reset_index()

total_score=total_score[total_score['innings']==1]
match_df=match.merge(total_score[['ID','total_run']],left_on='ID',right_on='ID')

match_df['Team1']=match_df['Team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['Team2']=match_df['Team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['Team1']=match_df['Team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['Team2']=match_df['Team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

delivery['BattingTeam']=delivery['BattingTeam'].str.replace('Delhi Daredevils','Delhi Capitals')

delivery['BattingTeam']=delivery['BattingTeam'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

teams= ['Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians', 'Kings XI Punjab']

match_df=match_df[match_df['Team1'].isin(teams)]
match_df=match_df[match_df['Team2'].isin(teams)]

delivery=delivery[delivery['BattingTeam'].isin(teams)]

match_df=match_df[['ID','City','WinningTeam','total_run','Team1','Team2']]
delivery_df= match_df.merge(delivery,on='ID')

delivery_df= delivery_df[delivery_df['innings']==2]
delivery_df['current_score'] = delivery_df.groupby('ID')['total_run_y'].transform('cumsum')
delivery_df['runs_left']= delivery_df['total_run_x']+1-delivery_df['current_score']
delivery_df['balls_left']=120-(delivery_df['overs']*6+delivery_df['ballnumber'])

delivery_df['wicket_left'] =10- delivery_df.groupby('ID')['isWicketDelivery'].transform('cumsum')
delivery_df['crr']=(delivery_df['current_score']/(delivery_df['overs']*6+delivery_df['ballnumber']))*6
delivery_df['req_rr']=(delivery_df['runs_left']/delivery_df['balls_left'])*6

def fun(row):
    return 1 if row['BattingTeam']==row['WinningTeam'] else 0

def bowl(row):
    return row['Team2'] if row['BattingTeam']==row['Team1'] else row['Team1']

delivery_df['result']=delivery_df.apply(fun,axis=1)
delivery_df['BowlingTeam']=delivery_df.apply(bowl,axis=1)

final_df=delivery_df[['BattingTeam','BowlingTeam','City','runs_left','balls_left','wicket_left','total_run_x','crr','req_rr','result']]
final_df.dropna(inplace=True)
final_df=final_df[final_df['balls_left']!=0]
final_df=final_df.sample(final_df.shape[0])

x=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1)

trf=  ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'),['BattingTeam','BowlingTeam','City'])],remainder='passthrough')

pipe=Pipeline(steps=[('step1',trf),('step2',LogisticRegression(solver='liblinear'))])
pipe.fit(X_train,Y_train)

pipe.predict_proba(X_test)[10]
#print(final_df.sample())
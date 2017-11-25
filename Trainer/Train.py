import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.externals import joblib

os.getcwd()

events = pd.read_json('Data/events.json')
groups = pd.read_json('Data/groups.json')
users = pd.read_json('Data/users.json')
venues = pd.read_json('Data/venues.json')

def parse_rsvp(rsvp_json):
    return sum([e['response']=='yes' for e in rsvp_json])

groups.loc[1]
users.loc[1]
venues.loc[1]
events.loc[2]

events['rsvp_yes'] = events.rsvps.apply(lambda x: parse_rsvp(x))
events['status'] == 'past'

selected_data = events[events.status == 'past'][['rsvp_yes','rsvp_limit']]
selected_data['rsvp_yes'].fillna(0, inplace=True)
selected_data['rsvp_limit'].fillna(0, inplace=True)
X = selected_data.drop('rsvp_yes', axis=1)
Y = selected_data['rsvp_yes']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

r2_train = r2_score(Y_train, rf.predict(X_train))
r2_test = r2_score(Y_test, rf.predict(X_test))
print("R-square train: %f" % r2_train)
print("R-square test: %f" % r2_test)

required_fields = ['rsvp_limit']

#Persist model and metadata
joblib.dump(rf, 'Data/rf.pkl')
joblib.dump(required_fields, 'Data/required_fields.pkl')


#to do:
"""
distribution of target
residual plots
create features
parsing class -> pipeline
switch to workbook
"""
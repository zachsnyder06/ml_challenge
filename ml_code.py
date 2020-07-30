# -*- coding: utf-8 -*-
"""
@author: zsnyd
"""

import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#read csv, create the target variable, and get the columns
df = pd.read_csv('Lending_Club_v2.csv')

bad_indicators =  ["Charged Off ","Charged Off","Default","Does not meet the credit policy. Status:Charged Off","In Grace Period","Default Receiver","Late (16-30 days)","Late (31-120 days)"]
df['target'] = df['loan_status'].isin(bad_indicators).astype(int)

cols = list(df.columns)
cols.sort()




#exploratory data analysis
df.loan_status.value_counts(dropna = False)
df.loan_status.value_counts(dropna = False, normalize = True)

#about 84% of loans were fully paid
df.target.value_counts(dropna = False, normalize = True)


#describe key variables
#lower average income for those that were charged off
df.groupby('target')['annual_inc'].describe().reset_index().transpose()

#interest rate
#higher average interest rate for charged off loans
df['int_rate_eda'] = df['int_rate'].str.strip('%').astype(float) #make float
df.groupby('target')['int_rate_eda'].describe().reset_index().transpose()


#debt to income
#slightly higher debt to income for charged off loans
df.groupby('target')['dti'].describe().reset_index().transpose()

#term
#longer average term for charged off loans
df_temp = df.copy(deep = True)
df_temp.term.dropna(inplace = True)
df_temp['term_eda'] = df_temp['term'].apply(lambda x: int(x.split()[0]))
df_temp.groupby('target')['term_eda'].describe().reset_index().transpose()
del df_temp



#drop variables used in eda
df.drop(columns = ['int_rate_eda'], inplace = True)


#feature reduction
'''
I'm getting rid of any variables that have more than 30% missing values.
Also, I dropped emp_title and title because the values are too granular,
and I dropped grade because the info is contained in sub_grade.

I also dropped each row with an NA.
'''
pctg_na = df.isna().mean().sort_values(ascending = False).reset_index()
vars_to_keep = pctg_na[pctg_na[0] < 0.3]

cols_to_keep = list(vars_to_keep['index'])
cols_to_keep.sort()


df2 = df[cols_to_keep]
df_view = df2.head(25)

pctg_unique = (df2.nunique()/len(df2.index)).reset_index()

#drop any record with na
df2.dropna(inplace = True)
df2.drop(columns = ['emp_title', 'grade', 'title'], inplace = True)


#feature selection
'''
After getting rid of variables with more than 30% missing.
I chose from the remaining 50 variables.
I used the Lending Club data dictionary to choose these variables.
My methodology was to exclude any variable that contained future information
about the loan.
I also looked at the importance of each variable used
after the model was trained (see end of script).
'''

x_vars = ['acc_now_delinq',
          'addr_state',
          'annual_inc',
          'application_type',
          'collections_12_mths_ex_med',
          'delinq_2yrs',
          'delinq_amnt',
          'dti',
          'earliest_cr_line',
          'emp_length',
          'hardship_flag',
          'home_ownership',
          'initial_list_status',
          'inq_last_6mths',
          'installment',
          'int_rate',
          'issue_d',
          'loan_amnt',
          'open_acc',
          'out_prncp',
          'pub_rec',
          'pub_rec_bankruptcies',
          'purpose',
          'pymnt_plan',
          'revol_util',
          'sub_grade',
          'term',
          'total_acc',
          'verification_status',
          'zip_code']

#select columns to be used in dummy variables function
dummy_cols = ['addr_state', 'application_type', 'emp_length',
              'hardship_flag', 'home_ownership', 'initial_list_status', 'purpose',
              'pymnt_plan', 'sub_grade', 'verification_status']


y = df2['target']
x = df2[x_vars]

x['zip_code'] = x['zip_code'].str.strip('xx').astype(int) #make zip code an integer
x['revol_util'] = x['revol_util'].str.strip('%').astype(float) #make float
x['int_rate'] = x['int_rate'].str.strip('%').astype(float) #make float
x['term'] = x['term'].apply(lambda x: int(x.split()[0])) #make integer
x.emp_length.replace('< 1 year', '0 years', inplace = True) #less than sign causes error


#get the year portion of the dates
x['earliest_cr_line'] = x['earliest_cr_line'].apply(lambda x: int(x.split('-')[1]))
x['issue_d'] = x['issue_d'].apply(lambda x: int(x.split('-')[1]))


#function to get the dummy variables for the categorical variables
x = pd.get_dummies(x, columns = dummy_cols, drop_first=True)


#6: build xgboost model
'''
This portion of the code builds a standard xgboost model.
I use the train_test_split function to create the test and training data.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 4)

model = XGBClassifier()
model.fit(x_train, y_train)



#7: show test and train score
'''
This part scores the model, finds the accuracy and the confusion matrix.
'''
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f'train score: {round(train_score, 4)}')
print(f'test score: {round(test_score, 4)}')

y_pred = model.predict(x_test)
accr_score = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)


'''
Here's the code that gets the importance of each variable used in the model.
'''
#print out importance of variables
#importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
importance = model.get_booster().get_score(importance_type = 'gain')

importance
len(importance.keys())





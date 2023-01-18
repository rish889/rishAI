import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

employee_df = pd.read_csv('./001-human-resource/Human_Resources.csv')
# pd.set_option('display.max_columns', None)
# print(employee_df.head(10))

# Let's replace the 'Attritition' and 'overtime' column with integers before performing any visualizations
employee_df['Attrition'] = employee_df['Attrition'].apply(
    lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(
    lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(
    lambda x: 1 if x == 'Y' else 0)

# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18',
                 'EmployeeNumber'], axis=1, inplace=True)

X_cat = employee_df[['BusinessTravel', 'Department',
                     'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

# note that we dropped the target 'Atrittion'
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',
                           'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',	'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
X_all = pd.concat([X_cat, X_numerical], axis=1)

scaler = MinMaxScaler()
X_all.columns = X_all.columns.astype(str)
X = scaler.fit_transform(X_all)

y = employee_df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\n\n############## LogisticRegression ################")
print("\nAccuracy : {} %".format(100 * accuracy_score(y_pred_lr, y_test)))
print("\nClassification Report : \n{} %".format(classification_report(y_test, y_pred_lr)))


model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("\n\n############## Random Forest ################")
print("\nAccuracy : {} %".format(100 * accuracy_score(y_pred_rf, y_test)))
print("\nClassification Report : \n{} %".format(classification_report(y_test, y_pred_rf)))

import pyodbc
from sqlalchemy import create_engine
import urllib
server = ''
database = ''
driver =  '{ODBC Driver 13 for SQL Server}'
Trusted_Connection = 'yes'
username = ''
password =''

def Bedrock(driver='{ODBC Driver 13 for SQL Server}',Trusted_Connection='yes',server = 'BedrockAD\\SQL',database = 'Bedrock_Client'):
    if Trusted_Connection == 'yes':
        return pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;auto_commit=True')
    else:
        return pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

def Red004(driver='{ODBC Driver 13 for SQL Server}',Trusted_Connection='yes',server = 'RBHDWHRED004',database = 'Bedrock_Client'):
    if Trusted_Connection == 'yes':
        return pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;auto_commit=True')
    else:
        return pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

def Red004_AA_engine(driver='{ODBC Driver 13 for SQL Server}',Trusted_Connection='yes',server = 'RBHDWHRED004',database = 'AdvancedAnalytics'):
    if Trusted_Connection == 'yes':
        quoted = urllib.parse.quote_plus('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes')
        return create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
    else:
        quoted = urllib.parse.quote_plus("'DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password")
        return create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))

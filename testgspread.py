import gspread
import pandas as pd 

# Create a pandas DataFrame
df = pd.DataFrame({'Name': ['Bob', 'Bill', 'David'], 
                   'Age': [25, 33, 49],
                   'Place': ['Bandung', 'Jakarta', 'Cirebon']})

# Authenticate with Google Sheets
gc = gspread.service_account(filename='C:/Users/keno/Downloads/trial1.json')

# Open the Google Sheets file
sh = gc.open('Test Sheets')

# Select the worksheet
worksheet = sh.sheet1

# Insert the DataFrame into the worksheet
worksheet.update([df.columns.values.tolist()] + df.values.tolist())
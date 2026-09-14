from flask import Flask, render_template, request, send_file
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__, template_folder='../Page 1')

# Initialize an empty dataframe
df = pd.DataFrame(columns=['Type', 'Brand', 'Color', 'Size', 'Rating'])

# Authenticate with Google Sheets
gc = gspread.service_account(filename='C:/Users/keno/Downloads/trial1.json')

# Open the Google Sheets file
sh = gc.open('Test Sheets')

# Select the worksheet
worksheet = sh.sheet1
worksheet.clear()

@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    if request.method == 'POST':
        # Get the input data from the form
        data = {
            'Type': request.form['question1'],
            'Brand': request.form['question2'],
            'Color': request.form['question3'],
            'Size': request.form['question4'],
            'Rating': request.form['question5']
        }
        # Append the data to the dataframe
        df = df.append(data, ignore_index=True)
        
        # Convert the dataframe to a list of lists
        data_list = df.values.tolist()
        
        # Insert the data into the Google Sheets worksheet
        worksheet.update([df.columns.values.tolist()] + data_list)
        
    return render_template('page1.htm', df=df.to_html(index=False, header=True, border=1))

@app.route('/image')
def display_image():
    return send_file('C:/Users/keno/Downloads/image-removebg-preview.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
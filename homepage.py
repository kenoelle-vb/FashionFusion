from flask import Flask, render_template
from flask import Flask, render_template, request, send_file
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from groq import Groq
import cv2
import cvzone
import os 
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import csv
import google.generativeai as genai


app = Flask(__name__, template_folder="../homepage", static_folder="C:/Users/keno/OneDrive/Documents/SGU Hackathon")

#Shit that needs to happen
chat_log = []
client = Groq(api_key="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")
df_wardrobe = pd.DataFrame(columns=['Type', 'Brand', 'Color', 'Size', 'Rating'])

apisheetskey = "1sIEI-_9N96ndRJgWDyl0iL65bACeGQ74MncOV4HQCXY"
url_apikey = f'https://docs.google.com/spreadsheet/ccc?key={apisheetskey}&output=csv'
df_apikey = pd.read_csv(url_apikey)

platform = "Gemini"
apikeyxloc = df_apikey['Platform'].str.contains(platform).idxmax()
apikey = df_apikey.iloc[apikeyxloc, 2]

genai.configure(api_key=apikey)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

contextsev = """
Available Items : 

Adidas : 
White Shirt, Average Rating=5, All sizes from : S, M, L, XL
White T-Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
White Sweater, Average Rating=4, All sizes from : S, M, L, XL
White Pants, Average Rating=4.5, All sizes from : S, M, L, XL
White Shoes, Average Rating=5, All sizes from : S, M, L, XL
Black Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Black T-Shirt, Average Rating=4, All sizes from : S, M, L, XL
Black Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
Black Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Black Shoes, Average Rating=5, All sizes from : S, M, L, XL
Red Shirt, Average Rating=4, All sizes from : S, M, L, XL
Red T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Red Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
Red Pants, Average Rating=5, All sizes from : S, M, L, XL
Red Shoes, Average Rating=4.5, All sizes from : S, M, L, XL
Green Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Green T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Green Sweater, Average Rating=5, All sizes from : S, M, L, XL
Green Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Green Shoes, Average Rating=4, All sizes from : S, M, L, XL
Blue Shirt, Average Rating=5, All sizes from : S, M, L, XL
Blue T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Blue Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
Blue Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Blue Shoes, Average Rating=4, All sizes from : S, M, L, XL
Yellow Shirt, Average Rating=4, All sizes from : S, M, L, XL
Yellow T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Sweater, Average Rating=5, All sizes from : S, M, L, XL
Yellow Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Shoes, Average Rating=5, All sizes from : S, M, L, XL

H&M :
White Shirt, Average Rating=5, All sizes from : S, M, L, XL
White T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
White Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
White Pants, Average Rating=3.5, All sizes from : S, M, L, XL
White Shoes, Average Rating=5, All sizes from : S, M, L, XL
Black Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
Black T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Black Sweater, Average Rating=5, All sizes from : S, M, L, XL
Black Pants, Average Rating=4, All sizes from : S, M, L, XL
Black Shoes, Average Rating=3.5, All sizes from : S, M, L, XL
Red Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
Red T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Red Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Red Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Red Shoes, Average Rating=4.5, All sizes from : S, M, L, XL
Green Shirt, Average Rating=5, All sizes from : S, M, L, XL
Green T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Green Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Green Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Green Shoes, Average Rating=4, All sizes from : S, M, L, XL
Blue Shirt, Average Rating=4, All sizes from : S, M, L, XL
Blue T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Blue Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Blue Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Blue Shoes, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Shirt, Average Rating=5, All sizes from : S, M, L, XL
Yellow T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Yellow Shoes, Average Rating=5, All sizes from : S, M, L, XL

Nike :
White Shirt, Average Rating=5, All sizes from : S, M, L, XL
White T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
White Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
White Pants, Average Rating=3.5, All sizes from : S, M, L, XL
White Shoes, Average Rating=5, All sizes from : S, M, L, XL
Black Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Black T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Black Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Black Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Black Shoes, Average Rating=5, All sizes from : S, M, L, XL
Red Shirt, Average Rating=5, All sizes from : S, M, L, XL
Red T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Red Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
Red Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Red Shoes, Average Rating=5, All sizes from : S, M, L, XL
Green Shirt, Average Rating=5, All sizes from : S, M, L, XL
Green T-Shirt, Average Rating=4, All sizes from : S, M, L, XL
Green Sweater, Average Rating=4.5, All sizes from : S, M, L, XL
Green Pants, Average Rating=4, All sizes from : S, M, L, XL
Green Shoes, Average Rating=5, All sizes from : S, M, L, XL
Blue Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Blue T-Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
Blue Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Blue Pants, Average Rating=5, All sizes from : S, M, L, XL
Blue Shoes, Average Rating=4, All sizes from : S, M, L, XL
Yellow Shirt, Average Rating=4, All sizes from : S, M, L, XL
Yellow T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Yellow Sweater, Average Rating=5, All sizes from : S, M, L, XL
Yellow Pants, Average Rating=5, All sizes from : S, M, L, XL
Yellow Shoes, Average Rating=3.5, All sizes from : S, M, L, XL

Zara :
White Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
White T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
White Sweater, Average Rating=4, All sizes from : S, M, L, XL
White Pants, Average Rating=4.5, All sizes from : S, M, L, XL
White Shoes, Average Rating=4.5, All sizes from : S, M, L, XL
Black Shirt, Average Rating=4, All sizes from : S, M, L, XL
Black T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Black Sweater, Average Rating=4, All sizes from : S, M, L, XL
Black Pants, Average Rating=4, All sizes from : S, M, L, XL
Black Shoes, Average Rating=5, All sizes from : S, M, L, XL
Red Shirt, Average Rating=4, All sizes from : S, M, L, XL
Red T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Red Sweater, Average Rating=3.5, All sizes from : S, M, L, XL
Red Pants, Average Rating=4.5, All sizes from : S, M, L, XL
Red Shoes, Average Rating=3.5, All sizes from : S, M, L, XL
Green Shirt, Average Rating=5, All sizes from : S, M, L, XL
Green T-Shirt, Average Rating=3.5, All sizes from : S, M, L, XL
Green Sweater, Average Rating=5, All sizes from : S, M, L, XL
Green Pants, Average Rating=3.5, All sizes from : S, M, L, XL
Green Shoes, Average Rating=5, All sizes from : S, M, L, XL
Blue Shirt, Average Rating=4.5, All sizes from : S, M, L, XL
Blue T-Shirt, Average Rating=4, All sizes from : S, M, L, XL
Blue Sweater, Average Rating=5, All sizes from : S, M, L, XL
Blue Pants, Average Rating=4, All sizes from : S, M, L, XL
Blue Shoes, Average Rating=5, All sizes from : S, M, L, XL
Yellow Shirt, Average Rating=4, All sizes from : S, M, L, XL
Yellow T-Shirt, Average Rating=5, All sizes from : S, M, L, XL
Yellow Sweater, Average Rating=4, All sizes from : S, M, L, XL
Yellow Pants, Average Rating=5, All sizes from : S, M, L, XL
Yellow Shoes, Average Rating=4, All sizes from : S, M, L, XL 
"""

trainsetkey = "1hSen30CQ8BaUKwsh7GjCbdjMbM1fjYTKLRgyriKYMB4"
url_trainset = f'https://docs.google.com/spreadsheet/ccc?key={trainsetkey}&output=csv'
df_trainset = pd.read_csv(url_trainset)
df_wardrobe = pd.read_csv(url_trainset)

df_trainset_length = df_trainset.shape[0]

new_df_userid = pd.DataFrame(index=range(df_trainset_length), columns=["User ID"])
new_df_userid["User ID"] = 101

new_df_productid = pd.DataFrame(index=range(df_trainset_length), columns=["Product ID"])
new_df_productid["Product ID"] = 999

combined_trainset_df = pd.concat([new_df_userid, new_df_productid, df_trainset], axis=1)

menkey = "1llW6H1KW_YVY_z-XSIgB6B0xi1Gh6gGiFsPeeFrlFi8"
url_men = f'https://docs.google.com/spreadsheet/ccc?key={menkey}&output=csv'
df_men = pd.read_csv(url_men)

df_fixed = pd.concat([df_men, combined_trainset_df], ignore_index=True)
df_fixed = df_fixed.drop(['Category', 'Price'], axis=1)

men_shirt_df = df_fixed[df_fixed["Type"].str.contains("Shirt", case=True)]
#men_shirt_df
men_shoes_df = df_fixed[df_fixed["Type"].str.contains("Shoes", case=False)]
#men_shoes_df
men_jeans_df = df_fixed[df_fixed["Type"].str.contains("Jeans", case=False)]
#men_jeans_df
men_tshirt_df = df_fixed[df_fixed["Type"].str.contains("T-shirt", case=False)]
#men_tshirt_df
men_sweater_df = df_fixed[df_fixed["Type"].str.contains("Sweater", case=False)]
#men_sweater_df

user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

men_shirt_df['User ID'] = user_encoder.fit_transform(men_shirt_df['User ID'])
men_shirt_df['Product ID'] = product_encoder.fit_transform(men_shirt_df['Product ID'])

men_shoes_df['User ID'] = user_encoder.fit_transform(men_shoes_df['User ID'])
men_shoes_df['Product ID'] = product_encoder.fit_transform(men_shoes_df['Product ID'])

men_jeans_df['User ID'] = user_encoder.fit_transform(men_jeans_df['User ID'])
men_jeans_df['Product ID'] = product_encoder.fit_transform(men_jeans_df['Product ID'])

men_tshirt_df['User ID'] = user_encoder.fit_transform(men_tshirt_df['User ID'])
men_tshirt_df['Product ID'] = product_encoder.fit_transform(men_tshirt_df['Product ID'])

men_sweater_df['User ID'] = user_encoder.fit_transform(men_sweater_df['User ID'])
men_sweater_df['Product ID'] = product_encoder.fit_transform(men_sweater_df['Product ID'])

train_df_shirt, test_df_shirt = train_test_split(men_shirt_df, test_size = 0.2)
train_df_shoes, test_df_shoes = train_test_split(men_shoes_df, test_size = 0.2)
train_df_jeans, test_df_jeans = train_test_split(men_jeans_df, test_size = 0.2)
train_df_tshirt, test_df_tshirt = train_test_split(men_tshirt_df, test_size = 0.2)
train_df_sweater, test_df_sweater = train_test_split(men_sweater_df, test_size = 0.2)

reader = Reader(rating_scale = (0.5, 5))

data_shirt = Dataset.load_from_df(train_df_shirt[['User ID', 'Product ID', 'Rating']], reader)
trainset_shirt = data_shirt.build_full_trainset()

data_shoes = Dataset.load_from_df(train_df_shoes[['User ID', 'Product ID', 'Rating']], reader)
trainset_shoes = data_shoes.build_full_trainset()

data_jeans = Dataset.load_from_df(train_df_jeans[['User ID', 'Product ID', 'Rating']], reader)
trainset_jeans = data_jeans.build_full_trainset()

data_tshirt = Dataset.load_from_df(train_df_tshirt[['User ID', 'Product ID', 'Rating']], reader)
trainset_tshirt = data_tshirt.build_full_trainset()

data_sweater = Dataset.load_from_df(train_df_sweater[['User ID', 'Product ID', 'Rating']], reader)
trainset_sweater = data_sweater.build_full_trainset()

model_svd = SVD()

model_svd.fit(trainset_shirt)
model_svd.fit(trainset_shoes)
model_svd.fit(trainset_jeans)
model_svd.fit(trainset_tshirt)
model_svd.fit(trainset_sweater)

predictions_svd_shirt = model_svd.test(trainset_shirt.build_anti_testset())
predictions_svd_shoes = model_svd.test(trainset_shoes.build_anti_testset())
predictions_svd_jeans = model_svd.test(trainset_jeans.build_anti_testset())
predictions_svd_tshirt = model_svd.test(trainset_tshirt.build_anti_testset())
predictions_svd_sweater = model_svd.test(trainset_sweater.build_anti_testset())

def get_top_n_recommendations_shirt(user_id, n=4):
    user_movies = men_shirt_df[men_shirt_df['User ID'] == user_id]['Product ID'].unique()
    all_movies = men_shirt_df['Product ID'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies_shirt = men_shirt_df[men_shirt_df['Product ID'].isin(top_n_movie_ids)][['Type', 'Brand', "Color"]]

    return top_n_movies_shirt

def get_top_n_recommendations_shoes(user_id, n=4):
    user_movies = men_shoes_df[men_shoes_df['User ID'] == user_id]['Product ID'].unique()
    all_movies = men_shoes_df['Product ID'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies_shoes = men_shoes_df[men_shoes_df['Product ID'].isin(top_n_movie_ids)][['Type', 'Brand', "Color"]]

    return top_n_movies_shoes

def get_top_n_recommendations_jeans(user_id, n=4):
    user_movies = men_jeans_df[men_jeans_df['User ID'] == user_id]['Product ID'].unique()
    all_movies = men_jeans_df['Product ID'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies_jeans = men_jeans_df[men_jeans_df['Product ID'].isin(top_n_movie_ids)][['Type', 'Brand', "Color"]]

    return top_n_movies_jeans

def get_top_n_recommendations_tshirt(user_id, n=4):
    user_movies = men_tshirt_df[men_tshirt_df['User ID'] == user_id]['Product ID'].unique()
    all_movies = men_tshirt_df['Product ID'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies_tshirt = men_tshirt_df[men_tshirt_df['Product ID'].isin(top_n_movie_ids)][['Type', 'Brand', "Color"]]

    return top_n_movies_tshirt

def get_top_n_recommendations_sweater(user_id, n=4):
    user_movies = men_sweater_df[men_sweater_df['User ID'] == user_id]['Product ID'].unique()
    all_movies = men_sweater_df['Product ID'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies_sweater = men_sweater_df[men_sweater_df['Product ID'].isin(top_n_movie_ids)][['Type', 'Brand', "Color"]]

    return top_n_movies_sweater

# =========================================== IMPORTANT USER ID =======================================================
user_id = 101
# =========================================== IMPORTANT USER ID =======================================================

recommendations_shirt = get_top_n_recommendations_shirt(user_id)
#print(f"Top 5 Recommendations for User {user_id}:")
#print(recommendations_shirt)

recommendations_shoes = get_top_n_recommendations_shoes(user_id)
#print(f"Top 5 Recommendations for User {user_id}:")
#print(recommendations_shoes)

recommendations_jeans = get_top_n_recommendations_jeans(user_id)
#print(f"Top 5 Recommendations for User {user_id}:")
#print(recommendations_jeans)

recommendations_sweater = get_top_n_recommendations_sweater(user_id)
#print(f"Top 5 Recommendations for User {user_id}:")
#print(recommendations_sweater)

recommendations_tshirt = get_top_n_recommendations_tshirt(user_id)
#print(f"Top 5 Recommendations for User {user_id}:")
#print(recommendations_tshirt)

shirt_color = recommendations_shirt.iloc[0, 2]
shirt_type = recommendations_shirt.iloc[0, 0]
shirtcell = shirt_color + shirt_type
#print(shirtcell)

shoes_color = recommendations_shoes.iloc[0, 2]
shoes_type = recommendations_shoes.iloc[0, 0]
shoescell = shoes_color + shoes_type
#print(shoescell)

jeans_color = recommendations_jeans.iloc[0, 2]
jeans_type = recommendations_jeans.iloc[0, 0]
jeanscell = jeans_color + jeans_type
#print(jeanscell)

tshirt_color = recommendations_tshirt.iloc[0, 2]
tshirt_type = recommendations_tshirt.iloc[0, 0]
tshirtcell = tshirt_color + tshirt_type
#print(tshirtcell)

sweater_color = recommendations_sweater.iloc[0, 2]
sweater_type = recommendations_sweater.iloc[0, 0]
sweatercell = sweater_color + sweater_type
#print(sweatercell)

shirt_index_key = "13L-pyPMLZ4vexfxoBrGeE26yh1cwW-YTd8ACDlTNuzI"
url_shirt_index = f'https://docs.google.com/spreadsheet/ccc?key={shirt_index_key}&output=csv'
df_clothes_index = pd.read_csv(url_shirt_index)
df_clothes_index

match_idx = df_clothes_index['Product Type'].str.contains(shirtcell).idxmax()
value_shirt = df_clothes_index.iloc[match_idx, 0]
#print(value_shirt)

match_idx = df_clothes_index['Product Type'].str.contains(shoescell).idxmax()
value_shoes = df_clothes_index.iloc[match_idx, 0]
#print(value_shoes)

match_idx = df_clothes_index['Product Type'].str.contains(jeanscell).idxmax()
value_jeans = df_clothes_index.iloc[match_idx, 0]
#print(value_jeans)

match_idx = df_clothes_index['Product Type'].str.contains(tshirtcell).idxmax()
value_tshirt = df_clothes_index.iloc[match_idx, 0]
#print(value_tshirt)

match_idx = df_clothes_index['Product Type'].str.contains(sweatercell).idxmax()
value_sweater = df_clothes_index.iloc[match_idx, 0]
#print(value_sweater)

shirt_color2 = recommendations_shirt.iloc[1, 2]
shirt_type2 = recommendations_shirt.iloc[1, 0]
shirtcell2 = shirt_color2 + shirt_type2
#print(shirtcell2)

shoes_color2 = recommendations_shoes.iloc[1, 2]
shoes_type2 = recommendations_shoes.iloc[1, 0]
shoescell2 = shoes_color2 + shoes_type2
#print(shoescell)

jeans_color2 = recommendations_jeans.iloc[1, 2]
jeans_type2 = recommendations_jeans.iloc[1, 0]
jeanscell2 = jeans_color2 + jeans_type2
#print(jeanscell)

tshirt_color2 = recommendations_tshirt.iloc[1, 2]
tshirt_type2 = recommendations_tshirt.iloc[1, 0]
tshirtcell2 = tshirt_color2 + tshirt_type2
#print(tshirtcell)

sweater_color2 = recommendations_sweater.iloc[1, 2]
sweater_type2 = recommendations_sweater.iloc[1, 0]
sweatercell2 = sweater_color2 + sweater_type2
#print(sweatercell)

match_idx = df_clothes_index['Product Type'].str.contains(shirtcell2).idxmax()
value_shirt2 = df_clothes_index.iloc[match_idx, 0]
#print(value_shirt2)

match_idx = df_clothes_index['Product Type'].str.contains(shoescell2).idxmax()
value_shoes2 = df_clothes_index.iloc[match_idx, 0]
#print(value_shoes2)

match_idx = df_clothes_index['Product Type'].str.contains(jeanscell2).idxmax()
value_jeans2 = df_clothes_index.iloc[match_idx, 0]
#print(value_jeans2)

match_idx = df_clothes_index['Product Type'].str.contains(tshirtcell2).idxmax()
value_tshirt2 = df_clothes_index.iloc[match_idx, 0]
#print(value_tshirt2)

match_idx = df_clothes_index['Product Type'].str.contains(sweatercell2).idxmax()
value_sweater2 = df_clothes_index.iloc[match_idx, 0]
#print(value_sweater2)

shirt_color3 = recommendations_shirt.iloc[2, 2]
shirt_type3 = recommendations_shirt.iloc[2, 0]
shirtcell3 = shirt_color3 + shirt_type3
#print(shirtcell3)

shoes_color3 = recommendations_shoes.iloc[2, 2]
shoes_type3 = recommendations_shoes.iloc[2, 0]
shoescell3 = shoes_color3 + shoes_type3
#print(shoescell3)

jeans_color3 = recommendations_jeans.iloc[2, 2]
jeans_type3 = recommendations_jeans.iloc[2, 0]
jeanscell3 = jeans_color3 + jeans_type3
#print(jeanscell3)

tshirt_color3 = recommendations_tshirt.iloc[2, 2]
tshirt_type3 = recommendations_tshirt.iloc[2, 0]
tshirtcell3 = tshirt_color3 + tshirt_type3
#print(tshirtcell3)

sweater_color3 = recommendations_sweater.iloc[2, 2]
sweater_type3 = recommendations_sweater.iloc[2, 0]
sweatercell3 = sweater_color3 + sweater_type3
#print(sweatercell3)

match_idx = df_clothes_index['Product Type'].str.contains(shirtcell3).idxmax()
value_shirt3 = df_clothes_index.iloc[match_idx, 0]
#print(value_shirt3)

match_idx = df_clothes_index['Product Type'].str.contains(shoescell3).idxmax()
value_shoes3 = df_clothes_index.iloc[match_idx, 0]
#print(value_shoes3)

match_idx = df_clothes_index['Product Type'].str.contains(jeanscell3).idxmax()
value_jeans3 = df_clothes_index.iloc[match_idx, 0]
#print(value_jeans3)

match_idx = df_clothes_index['Product Type'].str.contains(tshirtcell3).idxmax()
value_tshirt3 = df_clothes_index.iloc[match_idx, 0]
#print(value_tshirt3)

match_idx = df_clothes_index['Product Type'].str.contains(sweatercell3).idxmax()
value_sweater3 = df_clothes_index.iloc[match_idx, 0]
#print(value_sweater3)

shirt_color4 = recommendations_shirt.iloc[3, 2]
shirt_type4 = recommendations_shirt.iloc[3, 0]
shirtcell4 = shirt_color4 + shirt_type4
#print(shirtcell4)

shoes_color4 = recommendations_shoes.iloc[3, 2]
shoes_type4 = recommendations_shoes.iloc[3, 0]
shoescell4 = shoes_color4 + shoes_type4
#print(shoescell4)

jeans_color4 = recommendations_jeans.iloc[3, 2]
jeans_type4 = recommendations_jeans.iloc[3, 0]
jeanscell4 = jeans_color4 + jeans_type4
#print(jeanscell4)

tshirt_color4 = recommendations_tshirt.iloc[3, 2]
tshirt_type4 = recommendations_tshirt.iloc[3, 0]
tshirtcell4 = tshirt_color4 + tshirt_type4
#print(tshirtcell4)

sweater_color4 = recommendations_sweater.iloc[3, 2]
sweater_type4 = recommendations_sweater.iloc[3, 0]
sweatercell4 = sweater_color4 + sweater_type4
#print(sweatercell4)

match_idx = df_clothes_index['Product Type'].str.contains(shirtcell4).idxmax()
value_shirt4 = df_clothes_index.iloc[match_idx, 0]
#print(value_shirt4)

match_idx = df_clothes_index['Product Type'].str.contains(shoescell4).idxmax()
value_shoes4 = df_clothes_index.iloc[match_idx, 0]
#print(value_shoes4)

match_idx = df_clothes_index['Product Type'].str.contains(jeanscell4).idxmax()
value_jeans4 = df_clothes_index.iloc[match_idx, 0]
#print(value_jeans4)

match_idx = df_clothes_index['Product Type'].str.contains(tshirtcell4).idxmax()
value_tshirt4 = df_clothes_index.iloc[match_idx, 0]
#print(value_tshirt4)

match_idx = df_clothes_index['Product Type'].str.contains(sweatercell4).idxmax()
value_sweater4 = df_clothes_index.iloc[match_idx, 0]
#print(value_sweater4)

@app.route("/")
def home():
    return render_template("homepage.htm")

@app.route('/wardrobe', methods=['GET', 'POST'])
def wardrobe():
    global df_wardrobe
    # Authenticate with Google Sheets
    gc = gspread.service_account(filename='C:/Users/keno/Downloads/trial1.json')

    # Open the Google Sheets file
    sh = gc.open('Test Sheets')

    # Select the worksheet
    worksheet = sh.sheet1
    #worksheet.clear()
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
        df_wardrobe = df_wardrobe.append(data, ignore_index=True)

        # Convert the dataframe to a list of lists
        data_list = df_wardrobe.values.tolist()

        # Insert the data into the Google Sheets worksheet
        worksheet.update([df_wardrobe.columns.values.tolist()] + data_list)

    return render_template('wardrobe.htm', df_wardrobe=df_wardrobe.to_html(index=False, header=True, border=1))

@app.route("/vtryon")
def vtryon():
    return render_template("vtryon.htm")  # Replace 'vtryon.html' with your virtual try-on page templates

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    global chat_log
    global contextsev
    if request.method == "POST" :
        if 'user_input' in request.form:
            user_input = request.form["user_input"]
            question = f"You are an AI sales that answer's customer's question regarding the product. The maximum you can answer is one paragraph. Answer the question given on {user_input} based on the context of {contextsev}, and always add space and enter in new line for every line, make the answer as readable and tidy as possible. DO NOT GIVE AN INTRO AND DO NOT SAY WHERE YOU GOT THE INFORMATION FROM, JUST GIVE THE ANSWER. Act as a sales and persuade the person to buy, and the answers must be CONCISE and PERSUASIVE!!!"
            chat_session = model.start_chat(history=[])
            inputq = user_input
            questionpers = question
            response = chat_session.send_message(questionpers)
            answer = response.text
            answer = str(answer)
            answer_final = answer.replace("*", "\n")
            chat_log.append(user_input)
            chat_log.append(answer_final)
            return render_template("chatbot.htm", chat_log=chat_log)
    else :
        return render_template("chatbot.htm", chat_log=chat_log)

@app.route("/outfitrecommendation")
def outfit_recommendation():
    return render_template("outfitrecommendation.htm",
                           value_shirt=value_shirt,
                           value_shoes=value_shoes,
                           value_jeans=value_jeans,
                           value_tshirt=value_tshirt,
                           value_sweater=value_sweater)  # Replace 'outfitrecommendation.html' with your recommendation page template

@app.route("/or2")
def outfit_recommendation2():
    return render_template("or2.htm",
                           value_shirt2=value_shirt2,
                           value_shoes2=value_shoes2,
                           value_jeans2=value_jeans2,
                           value_tshirt2=value_tshirt2,
                           value_sweater2=value_sweater2)

@app.route("/or3")
def outfit_recommendation3():
    return render_template("or3.htm",
                           value_shirt3=value_shirt3,
                           value_shoes3=value_shoes3,
                           value_jeans3=value_jeans3,
                           value_tshirt3=value_tshirt3,
                           value_sweater3=value_sweater3)

@app.route("/or4")
def outfit_recommendation4():
    return render_template("or4.htm",
                           value_shirt4=value_shirt4,
                           value_shoes4=value_shoes4,
                           value_jeans4=value_jeans4,
                           value_tshirt4=value_tshirt4,
                           value_sweater4=value_sweater4)

if __name__ == "__main__":
    app.run(debug=True)

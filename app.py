import os
import flask
from flask import render_template, request, json
from flask import jsonify
import pandas as pd
import numpy as np
from flask import render_template, request, jsonify, session
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import hashlib
import json
from time import time

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):


        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions
        self.current_transactions = []

        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):

        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })

        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):

        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()


# Instantiate the Blockchain
blockchain = Blockchain()



app = flask.Flask(__name__, template_folder='Templates')
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
CORS(app)

book_rate_data = pd.read_csv("Data/training_data.csv")
ratings_books_nonzero = book_rate_data[book_rate_data['Book-Rating']!=0]

num_rating_df = ratings_books_nonzero.groupby('Book-Title').count()['Book-Rating'].sort_values(ascending=False).reset_index()
num_rating_df.rename(columns={'Book-Rating':'Number-of-Ratings'}, inplace=True)

avg_rating_df = ratings_books_nonzero.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)

popularity_df = pd.merge(num_rating_df, avg_rating_df, on='Book-Title')

popularity_df_above_100 = popularity_df[popularity_df['Number-of-Ratings']>=100]
popularity_df_above_50 = popularity_df[popularity_df['Number-of-Ratings'] >= 50]
popularity_df_above_250 = popularity_df[popularity_df['Number-of-Ratings'] >= 250]
popularity_df_above_100.sort_values(by='Average-Rating', ascending=False).head()

# Defining a new function that can calculate the metric
def calcWeightedRating(row, avgRating, numOfRatings, minThres, defRating):
    weightedRating = ((row[avgRating] * row[numOfRatings]) + (minThres * defRating))/(row[numOfRatings] + minThres)
    return weightedRating

# For number of ratings above 100
popularity_df_above_100 = popularity_df_above_100.copy()
popularity_df_above_100['Weighted-Rating'] = popularity_df_above_100.apply(lambda x: calcWeightedRating(
     x, 'Average-Rating', 'Number-of-Ratings', 100, 5),axis=1)
popularity_df_above_100.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 50
popularity_df_above_50 = popularity_df_above_50.copy()
popularity_df_above_50['Weighted-Rating'] = popularity_df_above_50.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 50, 5), axis=1)
popularity_df_above_50.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 250
popularity_df_above_250 = popularity_df_above_250.copy()
popularity_df_above_250['Weighted-Rating'] = popularity_df_above_250.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 250, 5), axis=1)
popularity_df_above_250.sort_values(
    'Weighted-Rating', ascending=False).head(20)

book_data = pd.read_csv("Data/Books.csv")

popular_df_merge = pd.merge(popularity_df_above_100, book_data, on='Book-Title').drop_duplicates('Book-Title',keep='first')
popular_df_merge = popular_df_merge.drop(columns=['Image-URL-S', 'Image-URL-L'])
popular_df_merge.sort_values('Weighted-Rating', ascending=False).head(10)

users_ratings_count = book_rate_data.groupby('User-ID').count()['ISBN']
users_ratings_count = users_ratings_count.sort_values(ascending=False).reset_index()
users_ratings_count.rename(columns={'ISBN':'No-of-Books-Rated'}, inplace=True)

users_200 = users_ratings_count[users_ratings_count['No-of-Books-Rated']>=200]

books_with_users_200 = pd.merge(users_200, book_rate_data, on='User-ID')

books_ratings_count = book_rate_data.groupby('Book-Title').count()['ISBN'].sort_values(ascending=False).reset_index()
books_ratings_count.rename(columns={'ISBN':'Number-of-Book-Ratings'}, inplace=True)

books_ratings_50 = books_ratings_count[books_ratings_count['Number-of-Book-Ratings']>=50]

filtered_books = pd.merge(books_ratings_50, books_with_users_200,  on='Book-Title')

books = pd.read_csv('Data/Books.csv')

famous_books = filtered_books.groupby('Book-Title').count().reset_index()
famous_books = famous_books['Book-Title']
famous_books = books[books['Book-Title'].isin(famous_books)]
famous_books = famous_books.copy()
famous_books.drop_duplicates(subset=['Book-Title'], inplace=True, keep='first')

pt = filtered_books.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

similarities = cosine_similarity(pt)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if flask.request.method == 'POST':
        
        
        book_name  = request.form['data'] 
        
        
       
        return_data = []
        bl = []
        if book_name in pt.index:
            index = np.where(pt.index == book_name)[0][0]
            similar_books_list = sorted(
            list(enumerate(similarities[index])), key=lambda x: x[1], reverse=True)[1:11]
            for book in similar_books_list:
                #return_data.append(pt.index[book[0]])
                bookname = pt.index[book[0]]
                # Get author and rating of similar book from DataFrame
                similar_book_info = book_rate_data[book_rate_data['Book-Title'] == bookname]
                
                # Extract author and rating
                similar_book_author = similar_book_info['Book-Author'].values[0]
                similar_book_rating = similar_book_info['Book-Rating'].values[0]
                similar_book_rating_arr = {
                    'Book-Title': bookname,
                    'Book-Author': similar_book_author,
                }
                bl.append(similar_book_rating_arr)
            rescount = len(bl)
            json_string = json.dumps(bl)
            des = {"d":"1"}
            return_data.append(des)
            return_data.append(json_string)
    
        else:
            filtered_books = book_rate_data[book_rate_data['Book-Title'].str.contains(book_name, case=False)]
            filtered_books_subset = filtered_books[['Book-Title', 'Book-Author']]
            json_array = filtered_books_subset.to_json(orient='records')
            rescount = len(json_array)
            des = {"d":"2"}
            return_data.append(des)
            return_data.append(json_array)
            
        search_data = {
            'query': book_name,
            'result_count': rescount
        }
        blockchain.new_transaction('me', 'recommenderblock', 1)
        block = blockchain.new_block(proof=12345)
        blc = json.dumps(blockchain.chain)
        return_data.append(blc)
        return jsonify(return_data)

@app.route('/')
@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

@app.route('/about', methods=['GET', 'POST'])
def about():
    if flask.request.method == 'GET':
        return(flask.render_template('about.html'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if flask.request.method == 'GET':
        return(flask.render_template('contact.html'))

@app.route('/recommendpage', methods=['GET', 'POST'])
def recommendpage():
    if flask.request.method == 'GET':
        return(flask.render_template('recommend.html'))

@app.route('/books', methods=['GET', 'POST'])
def books():
    if flask.request.method == 'GET':
        return(flask.render_template('books.html'))
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
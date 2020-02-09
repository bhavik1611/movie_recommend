from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
import numpy as np
import json
import pandas as pd
import numpy as np
import movie as mv

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')
gnomeT = pd.read_csv('genome-tags.csv')
gnomeS = pd.read_csv('genome-scores.csv')

app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
# CORS(app, resources={r"/api": {"origins": "*"}})

@app.route('/recommendations', methods=['GET'])
# @crossdomain(origin="*", headers=["Content-Type"])
def makecalc():
    userid = request.args.get('id')
    data = mv.getMovieRecommendation(int(userid),movies,ratings,links)	
    return data.to_json()


@app.route('/favourites', methods=['GET'])
# @crossdomain(origin="*", headers=["Content-Type"])
def getPercent():
    userid = request.args.get('id')
    data = mv.getFavGenre(int(userid),movies,ratings)	
    return data.to_json()

@app.route('/similarUsers', methods=['GET'])
# @crossdomain(origin="*", headers=["Content-Type"])
def getNeigh():
    userid = request.args.get('id')
    data = mv.getNeighbours(int(userid),movies,ratings)	
    return data[2].to_json()

@app.route('/likedBy', methods=['GET'])
# @crossdomain(origin="*", headers=["Content-Type"])
def similarMov():
    userid = request.args.get('id')
    data = mv.getSimilarMovies(int(userid),movies,ratings,links)	
    return data.to_json()

@app.route('/toprated',methods=['GET'])
# @crossdomain(origin="*", headers=["Content-Type"])
def topRated():
    n = request.args.get('n')
    data = mv.top_rated(movies,ratings, n)	
    return data.to_json()

if __name__ == '__main__':
    app.run(debug=True)

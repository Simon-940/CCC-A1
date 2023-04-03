from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

# load twitter data
with open('data/twitter-data-small.json', encoding='utf-8') as f:
    tweets = json.load(f)

# load location file
with open('data/sal.json', encoding='utf-8') as f:
    locations = json.load(f)

class Tweet():
    def __init__(self, tweet) -> None:
        self.set_tweet_data(tweet)
        
    def set_tweet_data(self,tweet):
        # extract place and author id from tweet json
        place = tweet['includes']['places'][0]['full_name'].lower()     
        matches = re.findall(r'\S+(?:\s+\S+)*(?=,)', place)    
        place = ''.join(matches)
        self.place = place
        
        self.author = tweet['_id']
    
    def get_place(self):
        return self.place
    
    def get_author(self):
        return self.author

capital_cities = {
    '1gsyd': 0,
    '2gmel': 0,
    '3gbri': 0,
    '4gade': 0,
    '5gper': 0,
    '6ghob': 0,
    '7gdar': 0,
    '8acte': 0,
    '9oter': 0,       
    }

irrelevant_areas = ['1rnsw', '2rvic', '3rqld', '4rsau', '5rwau', '6rtas', '7rnte']

tweeters = {}

for tweet in tweets:
    t = Tweet(tweet)
    city = t.get_place()
    id = t.get_author()
    # check if the place from tweet matches anything in the locations data. If not from irrelevant area and match, increment count
    if city in locations and locations[city]['gcc'] not in irrelevant_areas:
        gcc_code = locations[city]['gcc']
        capital_cities[gcc_code] += 1
        if id not in tweeters:
            # (# of tweets the user made, count for each city)
            tweeters[id] = (1, {gcc_code: 1})
        else:
            tweeters[id][0] += 1
            # add if city already exist
            if tweeters[id][1][gcc_code]:
                tweeters[id][1][gcc_code] += 1
            else:
                tweeters[id][1][gcc_code] = 1
                
    

# t1
top_cities = dict(sorted(capital_cities.items(), key=lambda item: item[1], reverse=True))
print(top_cities)

# t2
t2top_tweeters = sorted(tweeters.items(), key=lambda item: item[1][0], reverse=True)[:10]
print(t2top_tweeters)

# t3
t3top_tweeters = sorted(tweeters.items(), key=lambda item: len(item[1][1]), reverse=True)[:10]
print(t3top_tweeters)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ',rank)

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1)
elif rank == 1:
    data = comm.recv(source=0)
    print('On process 1, data is ',data)
import collections
import functools
import operator


import json
import re
from mpi4py import MPI

# load twitter data
with open('data/twitter-data-small.json', encoding='utf-8') as f:
    tweets = json.load(f)

# load location file
with open('data/sal.json', encoding='utf-8') as f:
    locations = json.load(f)


class Tweet():
    def __init__(self, tweet) -> None:
        self.set_tweet_data(tweet)

    def set_tweet_data(self, tweet):
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


def extract_data(tweets, locations, size, start_point):
    capital_cities = {
        '1gsyd': 0,
        '2gmel': 0,
        '3gbri': 0,
        '4gade': 0,
        '5gper': 0,
        '6ghob': 0,
        '7gdar': 0,
        '8acte': 0,
    }

    irrelevant_areas = ['1rnsw', '2rvic', '3rqld', '4rsau', '5rwau', '6rtas', '7rnte', '9oter']

    tweeters = {}

    for tweet in tweets[start_point:start_point + size]:
        t = Tweet(tweet)
        city = t.get_place()
        id = t.get_author()
        # check if the place from tweet matches anything in the locations data. If not from irrelevant area and
        # match, increment count
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

    return capital_cities, tweeters


# set MPI variable
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
# calculate chuck size using node size
chuck_size = len(tweets) // size
start_points = list(range(0, len(tweets), len(tweets) // size))[:size]
# scatter starting index of the chunk
start_point = comm.scatter(start_points, 0)
# run the main function
res = extract_data(tweets, locations, chuck_size, start_point)
# gather result
recv = comm.gather(res, 0)
# concatenate result
if rank == 0:
    # dict list for tasks
    cap_cities = [item[0] for item in recv]
    twt_users = [item[1] for item in recv]
    # sum values with same key
    res_city = dict(functools.reduce(operator.add, map(collections.Counter, cap_cities)))
    # TODO: task 2 and 3
    res_users = {}
    for node_dict in twt_users:
        for uid, data in node_dict.items():
            if uid in res_users:
                res_users[uid][0] += data[0]
                res_users[uid][1] = dict(functools.reduce(operator.add,
                                                          map(collections.Counter, [res_users[uid][1], data[1]])))
            else:
                res_users[uid] = data
    # t1
    top_cities = sorted(res_city.items(), key=lambda item: item[1], reverse=True)
    print(top_cities, "\n")

    # t2
    t2top_tweeters = sorted(res_users.items(), key=lambda item: item[1][0], reverse=True)[:10]
    print(t2top_tweeters, "\n")

    # t3
    t3top_tweeters = sorted(res_users.items(), key=lambda item: len(item[1][1]), reverse=True)[:10]
    print(t3top_tweeters, "\n")

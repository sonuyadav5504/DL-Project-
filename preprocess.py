import pandas as pd
import numpy as np
# import gc
import pickle as pkl
from tqdm import tqdm

class Dataset(object):
    def __init__(self, neg_size=99):
        ratings = []
        with open('ciao_rating_with_timestamp.txt') as f:
            for l in f:
                user, item, _, rating, _, timestamp = [int(float(_)) for _ in l.strip().split('  ')]
                ratings.append({
                    'user': user,
                    'item': item,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        item_count = ratings['item'].value_counts()
        item_count.name = 'item_count'
        ratings = ratings.join(item_count, on='item')
        self.ratings = ratings

        # test and validation set
        self.ratings['timerank'] = self.ratings.groupby('user')['timestamp'].rank(ascending=False).astype('int')
        self.ratings['test_mask'] = (self.ratings['timerank'] == 1)
        self.ratings['val_mask'] = (self.ratings['timerank'] == 2)

        # remove items that only appear in validation and test set
        items_selected = self.ratings[self.ratings['timerank'] > 2]['item'].unique()
        self.ratings = self.ratings[self.ratings['item'].isin(items_selected)].copy()
        users_selected = self.ratings[self.ratings['timerank'] > 2]['user'].unique()
        self.ratings = self.ratings[self.ratings['user'].isin(users_selected)].copy()

        # drop users and movies which do not exist in ratings
        self.users = self.ratings[['user']].drop_duplicates(subset=['user'],keep='first')
        self.items = self.ratings[['item']].drop_duplicates(subset=['item'],keep='first')
        self.rates = self.ratings[['rating']].drop_duplicates(subset=['rating'],keep='first')

        self.users_invmap = {u: i for i, u in enumerate(self.users['user'])}
        self.items_invmap = {m: i for i, m in enumerate(self.items['item'])}
        self.ratings_invmap = {r: i for i, r in enumerate(self.rates['rating'])}

        self.ratings['user_idx'] = self.ratings['user'].apply(lambda x: self.users_invmap[x])
        self.ratings['item_idx'] = self.ratings['item'].apply(lambda x: self.items_invmap[x])
        self.ratings['rating_idx'] = self.ratings['rating'].apply(lambda x: self.ratings_invmap[x])

        # parse item features
        self.num_items = len(self.items)
        self.num_users = len(self.users)

        # unobserved items for each user in training set
        self.neg_train = [None] * len(self.users)

        # negative examples for validation and test for evaluating ranking
        self.neg_val = np.zeros((len(self.users), neg_size), dtype='int64')
        self.neg_test = np.zeros((len(self.users), neg_size), dtype='int64')
        rating_groups = self.ratings.groupby('user_idx')


        for u in tqdm(range(len(self.users))):
            interacted_items = self.ratings['item_idx'].iloc[rating_groups.indices[u]]
            timerank = self.ratings['timerank'].iloc[rating_groups.indices[u]]

            interacted_items_val = interacted_items[timerank >= 1]
            neg_samples = np.setdiff1d(np.arange(len(self.items)), interacted_items_val)
            self.neg_val[u] = np.random.choice(neg_samples, neg_size)

            del neg_samples

            interacted_items_test = interacted_items[timerank >= 1]
            neg_samples = np.setdiff1d(np.arange(len(self.items)), interacted_items_test)
            self.neg_test[u] = np.random.choice(neg_samples, neg_size)

            del neg_samples

            # gc.collect()


if __name__ == "__main__":
    history_u, history_i, history_ur, history_ir = {}, {}, {}, {}
    data = Dataset()
    ratings = data.ratings
    ratings_train = ratings[~(ratings['val_mask'] | ratings['test_mask'])]
    user_idx = ratings_train['user_idx']
    item_idx = ratings_train['item_idx']
    rating_idx = ratings_train['rating_idx']
    ratings_set = set()

    for u, i, r in zip(user_idx, item_idx, rating_idx):
        ratings_set.add(r)
        if u not in history_u:
            history_u[u] = []
            history_ur[u] = []
        history_u[u].append(i)
        history_ur[u].append(r)

        if i not in history_i:
            history_i[i] = []
            history_ir[i] = []
        history_i[i].append(u)
        history_ir[i].append(r)

    assert data.num_users == len(history_u)
    assert data.num_items == len(history_i)

    train_u , train_i , train_r = [] , [] , []
    val_u , val_i , val_r = [] , [] , []
    test_u , test_i , test_r = [] , [] , []

    ratings_val = ratings[ratings['val_mask']]
    ratings_test = ratings[ratings['test_mask']]

    for u, i, r in zip(ratings_train['user_idx'] , ratings_train['item_idx'] , ratings_train['rating']):
        train_u.append(u)
        train_i.append(i)
        train_r.append(r)

    for u, i, r in zip(ratings_val['user_idx'] , ratings_val['item_idx'] , ratings_val['rating']):
        val_u.append(u)
        val_i.append(i)
        val_r.append(r)

    for u, i, r in zip(ratings_test['user_idx'] , ratings_test['item_idx'] , ratings_test['rating']):
        test_u.append(u)
        test_i.append(i)
        test_r.append(r)

    social_neighbor = {}
    num_neighbor = 0
    with open('ciao_trust.txt' , 'r' , encoding='utf-8') as f:
        for line in f:
            user1id , user2id = line.strip().split('  ')
            user1id , user2id = int(float(user1id)) , int(float(user2id))
            if user1id not in data.users_invmap or user2id not in data.users_invmap:
                continue
            user1id_idx , user2id_idx = data.users_invmap[user1id] , data.users_invmap[user2id]
            if user1id_idx not in social_neighbor:
                social_neighbor[user1id_idx] = []
            social_neighbor[user1id_idx].append(user2id_idx)
            num_neighbor += 1

    if len(social_neighbor) != data.num_users:
        print('# of users have friends: '+str(len(social_neighbor)))
        nofriends_users = np.setdiff1d(np.arange(data.num_users), list(social_neighbor.keys()))
        for u in nofriends_users:
            social_neighbor[u] = [u]
            num_neighbor += 1

    val_rank = {}
    test_rank = {}

    for u, i, r in zip(val_u , val_i , val_r):
        if r < 3:
            continue
        neg_samples = data.neg_val[u]
        val_rank[u] = {'pos': i , 'neg':neg_samples}

    for u, i, r in zip(test_u , test_i , test_r):
        if r < 3:
            continue
        neg_samples = data.neg_test[u]
        test_rank[u] = {'pos': i, 'neg':neg_samples}
        

    f = open('ciao_dataset.pkl', 'wb')
    data_content = (history_u, history_i, history_ur, history_ir, train_u, train_i, train_r, val_u, val_i, val_r,
                     test_u, test_i, test_r, social_neighbor, list(ratings_set))
    pkl.dump(data_content, f)
    f.close()

    f = open('ciao_dataset_rank.pkl', 'wb')
    pkl.dump((val_rank, test_rank), f)
    f.close()

    print('# of users: '+str(data.num_users))
    print('# of items: '+str(data.num_items))
    print('# of train interactions: '+str(len(train_u)))
    print('# of val interactions: '+str(len(val_u)))
    print('# of test interactions: '+str(len(test_u)))
    print('# of rating: '+str(len(list(ratings_set))))
    print('# of val rank interactions: '+str(len(val_rank)))
    print('# of test rank interactions: '+str(len(test_rank)))
    print('# of neighbors: '+str(num_neighbor))

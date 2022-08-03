# Install and load necesary packages

import pandas as pd
import numpy as np  

import warnings
warnings.filterwarnings("ignore")

## Load the dataset using pandas
df = pd.read_csv('ml-100k/u.data', names=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')

# obtain top 500 users and top 500 items

df.head()

# Split dataset
# remap user and item ID
df['user_id'] = df.groupby('user_id').ngroup()
df['item_id'] = df.groupby('item_id').ngroup()

test_df = df.groupby('user_id').sample(1, random_state=1024)
train_df = df[~df.index.isin(test_df.index)]

## Randomly select one rating from each user as test set
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
avg_num = df.groupby('user_id').size().mean()
density = df.shape[0] / (n_users * n_items)
min_ratings = df.rating.min()
max_ratings = df.rating.max()

print("The number of users: {}" .format(n_users))
print("The number of items: {}" .format(n_items))
print("Avg. # of rated Items/User: {}" .format(avg_num))
print("Density of data: {}" .format(density))
print("Ratings Range: {} - {}" .format(min_ratings, max_ratings))


# Convert the format of datasets to matrices
# Train dataset
df_zeros = pd.DataFrame({
    'user_id': np.tile(np.arange(0, n_users), n_items), 
    'item_id': np.repeat(np.arange(0, n_items), n_users), 
    'rating': 0})
train_ds = df_zeros.merge(train_df, 
                          how='left', 
                          on=['user_id', 'item_id']).fillna(0.).pivot_table(
                              values='rating_y', 
                              index='user_id', 
                              columns='item_id').values
                           
# Test dataset
test_ds = df_zeros.merge(test_df, 
                         how='left', 
                         on=['user_id', 'item_id']).fillna(0.).pivot_table(
                             values='rating_y', 
                             index='user_id', 
                             columns='item_id').values

print("Construct the rating matrix based on train_df:")
print(train_ds)

print("Construct the rating matrix based on test_df:")
print(test_ds)


############################## Utils

EPSILON = 1e-9    # handle the 0 denominator

def user_corr(imputed_train_ds):
    '''
    Function for calculating user's similarity
    '''
    active_user_pearson_corr = np.zeros((imputed_train_ds.shape[0], imputed_train_ds.shape[0]))

    # Compute Pearson Correlation Coefficient of All Pairs of Users between active set and training dataset
    for i, user_i_vec in enumerate(imputed_train_ds):
        for j, user_j_vec in enumerate(imputed_train_ds):

            # ratings corated by the current pair od users
            mask_i = (user_i_vec > 0)
            mask_j = (user_j_vec > 0)

            # corrated item index, skip if there are no corrated ratings
            corrated_index = np.intersect1d(np.where(mask_i), np.where(mask_j))   #### locate the existing correlated pairs
            if len(corrated_index) == 0:
                continue

            # average value of user_i_vec and user_j_vec
            mean_user_i = np.sum(user_i_vec) / (np.sum(np.clip(user_i_vec, 0, 1)) + EPSILON)  ### effective row average
            mean_user_j = np.sum(user_j_vec) / (np.sum(np.clip(user_j_vec, 0, 1)) + EPSILON)  ### effective row average

            p = pt[:,1]
            w = log(m/p)
            
            items  n_user    b   c
            1      50      50  50
            2      49      49  49
            3      20      20  20

            # compute pearson corr
            user_i_sub_mean = user_i_vec[corrated_index] - mean_user_i
            user_j_sub_mean = user_j_vec[corrated_index] - mean_user_j

            r_ui_sub_r_i_sq = np.square(user_i_sub_mean)
            r_uj_sub_r_j_sq = np.square(user_j_sub_mean)

            r_ui_sum_sqrt = np.sqrt(np.sum(r_ui_sub_r_i_sq))
            r_uj_sum_sqrt = np.sqrt(np.sum(r_uj_sub_r_j_sq))

            sim = np.sum(w**2 * user_i_sub_mean * user_j_sub_mean) / (r_ui_sum_sqrt * r_uj_sum_sqrt + EPSILON)  # equ(2)
            active_user_pearson_corr[i][j] = sim

    return active_user_pearson_corr

def predict(test_ds, imputed_train_ds, user_corr, k=20):
    '''
    Function for predicting ratings in test_ds
    '''

    # Predicting ratings of test set
    predicted_ds = np.zeros_like(test_ds)

    for (i, j), rating in np.ndenumerate(test_ds):

        if rating > 0:

            # only predict ratings on test set
            sim_user_ids = np.argsort(user_corr[i])[-1:-(k + 1):-1]           ## find the largest k individuals in ith row

            #==================user-based==================#
            # the coefficient values of similar users
            sim_val = user_corr[i][sim_user_ids]           

            # the average value of the current user's ratings
            sim_users = imputed_train_ds[sim_user_ids]
            
            mask_rateditem_user = (imputed_train_ds[i] != 0)  
            num_rated_items = mask_rateditem_user.astype(np.float32)          # turn boolean (0 or 1) into float
            user_mean = np.sum(imputed_train_ds[i, mask_rateditem_user]) / (num_rated_items.sum() + EPSILON)   

            mask_nei_rated_items = sim_users != 0        # largest k individuals could be 0 so we need to remove them
            num_rated_per_user = mask_nei_rated_items.astype(np.float32)  # turn boolean (0 or 1) into float
            num_per_user = num_rated_per_user.sum(axis=1)    # sum the ones to find the number of valid neighbour

            sum_per_user = sim_users.sum(axis=1) # axis =1  -> row sum
            sim_user_mean = sum_per_user / (num_per_user + EPSILON)
            
            mask_rated_j = (sim_users[:, j] > 0)
                            
            # sim(u, v) * (r_vj - mean_v)
            sim_r_sum_mean = sim_val[mask_rated_j] * (sim_users[mask_rated_j, j] - sim_user_mean[mask_rated_j])
            
            user_based_pred = user_mean + np.sum(sim_r_sum_mean) / (np.sum(sim_val[mask_rated_j]) + EPSILON)    # equ (3)

            predicted_ds[i, j] = np.clip(user_based_pred, 0, 5)   #limited by 0 and 5
            
    return predicted_ds

def evaluate(test_ds, predicted_ds):
    '''
    Function for evaluating on MAE and RMSE
    '''
    # MAE
    mask_test_ds = (test_ds > 0)
    MAE = np.sum(np.abs(test_ds[mask_test_ds] - predicted_ds[mask_test_ds])) / np.sum(mask_test_ds.astype(np.float32))

    # RMSE
    RMSE = np.sqrt(np.sum(np.square(test_ds[mask_test_ds] - predicted_ds[mask_test_ds])) / np.sum(mask_test_ds.astype(np.float32)))

    return MAE, RMSE
    
# Baseline - KNN based recommendation (Similarity Metric: Pearson Correlation Coefficient)
 
user_pearson_corr = user_corr(train_ds)
predicted_ds = predict(test_ds, train_ds, user_pearson_corr, k=20)

MAE, RMSE = evaluate(test_ds, predicted_ds)

print("===================== Baseline Result =====================")
print("MAE: {}, RMSE: {}" .format(MAE, RMSE))

import pickle
import numpy as np
a_path = '/media/liuxz/comma/sentiment_analysis_datasets/mosei/cmu_mosei/seq_length_50/mosei_senti_data.pkl'
with open(a_path,'rb') as f:
    a_data = pickle.load(f)
    # np.array(a_data)
print(a_data.shape)
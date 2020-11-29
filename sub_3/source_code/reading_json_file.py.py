import json
import os
import random
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

# The original dataset is 50 videos of pushing, and 50 videos of other activity.
# The reason to separate videos is to facilitate data preparation and avoid manual labeling.
# openpose is used to get landmark features of each video frame as json file
# To keep track of all frames of the videos including their landmark features and coresponding labels, a dictionary is defined
# where keys are the label of each video (push_0, push_1, ..., other_0, other_1, ..) and values are vector of size (n,75)
# where 'n' is the number of frames and 75 is fixed for all json files which represent the vector size of the land mark: 'pose_keypoints_2d'


# input = push_0_000000000000_keypoints
# output= push_0
def get_video_name(file_name):
    parts = file_name.split('_')
    return (parts[0] + '_' + parts[1])


# the if statement is added since some json files did not have 'pose_keypoints_2d'

# returning a dict where keys are video labels(push_0,push_1, ..)
# and values are landmark vectors corresponding to frames of the video
def read_data_from_landmarks(json_dir):
    video_to_landmarks = {}
    for file in os.listdir(json_dir):
        file_path = json_dir + '\\' + file
        temp = json.load(open(file_path))
        if len(temp['people']) == 0:
            pass
        else:
            key = get_video_name(file)
            value = temp['people'][0]['pose_keypoints_2d']
            video_to_landmarks.setdefault(key, []).append(value)

    return video_to_landmarks

# dictonary is a hashtable  and shuffle not work.To shuffle :
# get the list of keys, shuffle the keys and get the values of the corresponding keys
# define a seed for random to be iterable --> random.Random(seed)
def shuffle_data (video_to_landmarks):
    shuffled_keys = list(video_to_landmarks.keys())
    random.Random(4).shuffle(shuffled_keys)
    labels = shuffled_keys
    train_data = []
    for key in labels:
        train_data.append(video_to_landmarks[key])

    return train_data, labels

# After this step, No SHUFFLE should be applied in any other step

# Driver
json_dir = "D:\\tamu\\courses\\DeepLearning\\ProjectPart5\\openpose_json"
video_to_landmarks = read_data_from_landmarks(json_dir)
train_data, labels = shuffle_data(video_to_landmarks)

# padding frames of different videos, the bellow function will automatically pad to max length
input_data = pad_sequences(train_data, dtype='float32', maxlen = 125, padding='post')

# get dataframe so the data could be saved as CSV file as input: (dataframe input should be 2d)
# reshaping the data so it could convertto dataframe
r = input_data.shape[0]
m = input_data.shape[1]
n = input_data.shape[2]

reshaped_inputdata = input_data.reshape(r, m*n)
input_df = pd.DataFrame(data=reshaped_inputdata, index=labels)
csv_filepath = "D:\\tamu\\courses\\DeepLearning\\ProjectPart5\\input_data_2.csv"
input_df.to_csv(csv_filepath)


print('input_data.shape: ',input_data.shape)
print('reshaped_inputdata.shape: ', reshaped_inputdata.shape)
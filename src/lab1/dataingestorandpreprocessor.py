import numpy as np
import os
from skimage import io
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random

base_path = './lab1data/'
base_img_path = base_path + 'data_256/'

mame_toy_dataset = pd.read_csv(base_path + 'MAMe_dataset.csv')

mame_toy_train_imgfiles = mame_toy_dataset[mame_toy_dataset['Subset'] == 'train']['Image file']
mame_toy_val_imgfiles = mame_toy_dataset[mame_toy_dataset['Subset'] == 'val']['Image file']
mame_toy_test_imgfiles = mame_toy_dataset[mame_toy_dataset['Subset'] == 'test']['Image file']

# reading image lab1data for the first time
mame_toy_train_imgs = [io.imread(os.path.join(base_img_path, fname)) for fname in mame_toy_train_imgfiles]
mame_toy_val_imgs = [io.imread(os.path.join(base_img_path, fname)) for fname in mame_toy_val_imgfiles]
mame_toy_test_imgs = [io.imread(os.path.join(base_img_path, fname)) for fname in mame_toy_test_imgfiles]


# Saving as pkls for faster subsequent reads
train_imgs_pkl = 'pkls/train_imgs.pkl'
with open(base_path + train_imgs_pkl, 'wb') as f:
    pickle.dump(mame_toy_train_imgs, f)
val_imgs_pkl = 'pkls/val_imgs.pkl'
with open(base_path + val_imgs_pkl, 'wb') as f:
    pickle.dump(mame_toy_val_imgs, f)
test_imgs_pkl = 'pkls/test_imgs.pkl'
with open(base_path + test_imgs_pkl, 'wb') as f:
    pickle.dump(mame_toy_test_imgs, f)


# If its not a first-time read from individual files, read image data from previously saved pkls
# with open(base_path + 'pkls/train_imgs.pkl', 'rb') as f:
#     mame_toy_train_imgs =  pickle.load(f)
# with open(base_path + 'pkls/val_imgs.pkl', 'rb') as f:
#     mame_toy_val_imgs =  pickle.load(f)
# with open(base_path + 'pkls/test_imgs.pkl', 'rb') as f:
#     mame_toy_test_imgs =  pickle.load(f)

###### Visualising data and its distribution ###########
classes = mame_toy_dataset['Medium'].unique()
subsets = ['train', 'val', 'test']
x = np.arange(len(classes))
fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(15)
plt.bar(x-0.2, height=mame_toy_dataset[mame_toy_dataset['Subset'] == 'train']['Medium'].value_counts(), width=0.2)
plt.bar(x, height=mame_toy_dataset[mame_toy_dataset['Subset'] == 'val']['Medium'].value_counts(), width=0.2)
plt.bar(x+0.2, height=mame_toy_dataset[mame_toy_dataset['Subset'] == 'test']['Medium'].value_counts(), width=0.2)
plt.xlabel('Categories')
plt.ylabel('# of images')
plt.xticks(x, classes)
plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.legend(['Train', 'Val', 'Test'])
plt.savefig(base_path + 'dataex/train_val_test_category_split.pdf')
plt.close()

# images from the train subset, one from each category, just to make sure reads are successful
fig, ax = plt.subplots(5, 6)
fig.set_figheight(25)
fig.set_figwidth(30)

train = mame_toy_dataset[mame_toy_dataset['Subset'] == 'train']

ctr = 0
for i in range(5):
  for j in range(6):
    if ctr < 29:
      ax[i][j].imshow(io.imread(os.path.join(base_img_path, train[train['Medium'] == classes[ctr]].iloc[0]['Image file'])))
      ax[i][j].set_title(classes[ctr])
      ax[i][j].axis('off')
      ctr += 1
plt.axis('off')
plt.savefig(base_path + 'dataex/representative_images.pdf')
plt.close()


########## Preprocessing lab1data ###########
# Normalising the images
mame_toy_train_imgs = np.array(mame_toy_train_imgs, dtype=np.float32) / 255
mame_toy_val_imgs = np.array(mame_toy_val_imgs, dtype=np.float32) / 255
mame_toy_test_imgs = np.array(mame_toy_test_imgs, dtype=np.float32) / 255



# preparing the labels as OHE

le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df['Medium'] = le.fit_transform(mame_toy_dataset[mame_toy_dataset['Subset'] == 'train']['Medium'])
val_df['Medium'] = le.fit_transform(mame_toy_dataset[mame_toy_dataset['Subset'] == 'val']['Medium'])
test_df['Medium'] = le.fit_transform(mame_toy_dataset[mame_toy_dataset['Subset'] == 'test']['Medium'])

mame_toy_train_labels = pd.DataFrame(ohe.fit_transform(train_df[['Medium']]).toarray()).values
mame_toy_val_labels = pd.DataFrame(ohe.fit_transform(val_df[['Medium']]).toarray()).values
mame_toy_test_labels = pd.DataFrame(ohe.fit_transform(test_df[['Medium']]).toarray()).values

print('Train images shape: ' + str(mame_toy_train_imgs.shape))
print('Val images shape: ' + str(mame_toy_val_imgs.shape))
print('Test images shape: ' + str(mame_toy_test_imgs.shape))
print('Train labels shape: ' + str(mame_toy_train_labels.shape))
print('Val labels shape: ' + str(mame_toy_val_labels.shape))
print('Test labels shape: ' + str(mame_toy_test_labels.shape))
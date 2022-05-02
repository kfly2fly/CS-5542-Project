import glob
import os
import random
import shutil
import numpy as np

import glob
import os
import shutil

#new might use this one
def create_kfold(path, k):
    adj_path = os.path.join(path, 'full/raw/adj')
    print(adj_path)
    labels = os.listdir(adj_path)
    for label in labels:
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/train/raw/adj/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/train/raw/graphlabel/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/train/raw/nvocabf/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/validation/raw/adj/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/validation/raw/graphlabel/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/validation/raw/nvocabf/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/test/raw/adj/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/test/raw/graphlabel/' + label))
        os.makedirs(os.path.join(path, 'kfold_' + str(k) + '/test/raw/nvocabf/' + label))


    adj_files = [f for f in glob.glob(os.path.join(adj_path, '*/*.adj'), recursive=True)]
    graphlabel_files = [f.replace('adj', 'graphlabel', 1).replace('.adj', '.gf', 1) for f in adj_files]
    node_feature_files = [f.replace('adj', 'nvocabf', 1).replace('.adj', '.ndf', 1) for f in adj_files]
    train_index = list(range(0, len(adj_files)))
    test_index = [(i * 10 + random.randint(0, 9)) for i in range(0, int(len(adj_files) / 10))]
    for a in test_index:
        train_index.remove(a)
    val_index = [train_index[i * 10 + random.randint(0, 9)] for i in range(0, int(len(train_index) / 10))]
    for a in val_index:
        train_index.remove(a)
        
    with open(os.path.join(path, 'kfold_' + str(k) +'/train_index.txt'), 'w') as w:
        for i in train_index:
            w.writelines(str(i) + '\n')
    with open(os.path.join(path, 'kfold_' + str(k) +'/val_index.txt'), 'w') as w:
        for i in val_index:
            w.writelines(str(i) + '\n')
    with open(os.path.join(path, 'kfold_' + str(k) +'/test_index.txt'), 'w') as w:
        for i in test_index:
            w.writelines(str(i) + '\n')
    for i, file in enumerate(adj_files):
        if i in test_index:
            shutil.copy(file, file.replace('full', 'kfold_' + str(k) + '/test'))
            shutil.copy(graphlabel_files[i], graphlabel_files[i].replace('full', 'kfold_' + str(k) + '/test'))
            shutil.copy(node_feature_files[i], node_feature_files[i].replace('full', 'kfold_' + str(k) + '/test'))
        elif i in val_index:
            shutil.copy(file, file.replace('full', 'kfold_' + str(k) + '/validation'))
            shutil.copy(graphlabel_files[i], graphlabel_files[i].replace('full', 'kfold_' + str(k) + '/validation'))
            shutil.copy(node_feature_files[i], node_feature_files[i].replace('full', 'kfold_' + str(k) + '/validation'))
        else:
            shutil.copy(file, file.replace('full', 'kfold_' + str(k) + '/train'))
            shutil.copy(graphlabel_files[i], graphlabel_files[i].replace('full', 'kfold_' + str(k) + '/train'))
            shutil.copy(node_feature_files[i], node_feature_files[i].replace('full', 'kfold_' + str(k) + '/train'))


create_kfold('E:\\project\\multi_final',23)
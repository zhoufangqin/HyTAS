import torch
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
from .hsi_augmentation import hsi_aug
from sklearn.decomposition import PCA
import random

## sample_gt function taken from https://github.com/szubing/S-DMM/blob/master/tools.py
def sample_gt(gt, train_size, mode='fixed_withone'):
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
        if mode == 'random':
            train_size = float(train_size) / 100

    if mode == 'random_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            train_len = int(np.ceil(train_size * len(X)))
            train_indices += random.sample(X, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0

    elif mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
#-------------------------------------------------------------------------------

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes): #16
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2 #if patch=5 padding=2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):#(?)
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)

    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3, is_train=False):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    if is_train:
        x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
        # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
        for j in range(test_point.shape[0]):
            x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
        # for k in range(true_point.shape[0]):
        #     x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
        print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
        # print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    if is_train:
        x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
        # x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
        print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
        # print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")

    if is_train:
        # return x_train_band, x_test_band, x_true_band, x_train, x_test, x_true
        return x_train_band, x_test_band, x_train, x_test
    else:
        return x_train_band, x_train
#-------------------------------------------------------------------------------

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    # for i in range(num_classes+1):
    #     for j in range(number_true[i]):
    #         y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    # print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    # return y_train, y_test, y_true
    return y_train, y_test

## randomly take few training samples from each category
def take_few_samples_each(train_data, train_label, num_classes, num_samples=1):
    for i in range(num_classes):
        x_tmp = train_data[train_label==i]
        y_tmp = train_label[train_label==i]
        indx = torch.randint(0, x_tmp.shape[0], (num_samples,))
        x_tmp = x_tmp[indx]
        y_tmp = y_tmp[indx]
        if i == 0:
            x_train_few = x_tmp
            y_train_few = y_tmp
        else:
            x_train_few = torch.cat([x_train_few, x_tmp], dim=0)
            y_train_few = torch.cat([y_train_few, y_tmp], dim=0)
    return  x_train_few, y_train_few

## randomly take one training sample from each category
def take_one_sample_each(train_data, train_label, num_classes):
    for i in range(num_classes):
        x_tmp = train_data[train_label==i]
        y_tmp = train_label[train_label==i]
        indx = torch.randint(0, x_tmp.shape[0], (1,))
        x_tmp = x_tmp[indx]
        y_tmp = y_tmp[indx]
        if i == 0:
            x_train_one = x_tmp
            y_train_one = y_tmp
        else:
            x_train_one = torch.cat([x_train_one, x_tmp], dim=0)
            y_train_one = torch.cat([y_train_one, y_tmp], dim=0)
    return  x_train_one, y_train_one

def HSI_data(is_train, data_set, in_chans=1):
    # prepare data
    if data_set == 'Indian':
        data = loadmat('./dataset/IndianPine.mat')
    elif data_set == 'Houston':
        data = loadmat('./dataset/Houston.mat')
    elif data_set == 'Pavia':
        data = loadmat('./dataset/Pavia.mat')
    elif data_set == 'KSC': #total 5211 valid samples, 13 classes
        data = {}
        img = loadmat('./dataset/KSC_corrected.mat')['KSC']
        gt = loadmat('./dataset/KSC_gt.mat')['KSC_gt']
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print(
                "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
        img[nan_mask] = 0
        gt[nan_mask] = 0
        # # Normalization
        # img = np.asarray(img, dtype='float32')
        # n_bands = img.shape[-1]
        # for band in range(n_bands):
        #     min_val = np.min(img[:, :, band])
        #     max_val = np.max(img[:, :, band])
        #     img[:, :, band] = (img[:, :, band] - min_val) / (max_val - min_val)
        train_gt, test_gt = sample_gt(gt, 25) #take 25 samples each class for training, use train_batch_size=32 as if using 64 and train with distributed 5 nodes causes no enough sample problem
        data['input'] = img
        data['TR'] = train_gt
        data['TE'] = test_gt
    elif data_set == 'Salinas': ##total 54129 valid samples, 16 classes
        data = {}
        img = loadmat('./dataset/Salinas_corrected.mat')['salinas_corrected']
        gt = loadmat('./dataset/Salinas_gt.mat')['salinas_gt']
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print(
                "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
        img[nan_mask] = 0
        gt[nan_mask] = 0
        train_gt, test_gt = sample_gt(gt,50)  # take 50 samples each class for training
        data['input'] = img
        data['TR'] = train_gt
        data['TE'] = test_gt
    else:
        print('Wrong dataset name, please check.')
        return

    TR = data['TR'] # train dataset
    TE = data['TE'] # test dataset
    input = data['input'] #(145,145,200)
    # input, _ = applyPCA(input, int(input.shape[2] * 0.6))

    label = TR + TE # content in the blank: label(1-16)
    num_classes = np.max(TR)

    # normalize data by band norm
    input_normalize = np.zeros(input.shape) #(145,145,200)
    for i in range(input.shape[2]): # min-max normalization
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    # data size
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=7)
    ## band_patch=3 for SpectralFormer, band_patch=1 for HyLITE
    if is_train:
        # x_train_band, x_test_band, x_true_band, x_train_, x_test_, x_true_ = train_and_test_data(mirror_image, band,
        #                                                                                          total_pos_train,
        #                                                                                          total_pos_test,
        #                                                                                          total_pos_true,
        #                                                                                          patch=7,
        #                                                                                          band_patch=in_chans, is_train=is_train)
        x_train_band, x_test_band, x_train_, x_test_ = train_and_test_data(mirror_image, band,
                                                                                                 total_pos_train,
                                                                                                 total_pos_test,
                                                                                                 total_pos_true,
                                                                                                 patch=7,
                                                                                                 band_patch=in_chans,
                                                                                                 is_train=is_train)
    else:
        x_train_band, x_train_ = train_and_test_data(mirror_image, band,
                                                         total_pos_train,
                                                         total_pos_test,
                                                         total_pos_true,
                                                         patch=7,
                                                         band_patch=in_chans, is_train=is_train)
    # y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    y_train, y_test = train_and_test_label(number_train, number_test, number_true, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [695, 200, 49]
    y_train = torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    apply_aug = False
    take_fewshot = True
    if apply_aug:
        if take_fewshot:
            # ## take one training sample from each category
            # x_train_one, y_train_one = take_one_sample_each(x_train, y_train, num_classes)
            # # print(x_train_one.size()) #16,200,49
            # x_train_one = x_train_one.view(x_train_one.size(0), x_train_one.size(1), int(x_train_one.size(2)**(1/2)),-1)
            # Label_train_hsi = hsi_aug(x_train_one, y_train_one)

            ## apply Gaussian noise
            x_train_one, y_train_one = take_few_samples_each(x_train, y_train, num_classes, num_samples=1)
            noise = torch.randn_like(x_train_one.permute(0,2,1)) * 0.1
            x_train_one = x_train_one + noise.permute(0,2,1)
            Label_train_hsi = Data.TensorDataset(x_train_one, y_train_one)
        else:
            x_train_ = torch.from_numpy(x_train_).type(torch.FloatTensor).permute(0, 3, 1, 2)  # [695, 200, 7, 7]
            ## apply augmentation
            Label_train_hsi = hsi_aug(x_train_, y_train)
    else:
        if take_fewshot:
            ## take one training sample from each category
            x_train_one, y_train_one = take_few_samples_each(x_train, y_train, num_classes, num_samples=1)
            Label_train_hsi=Data.TensorDataset(x_train_one,y_train_one)
        else:
            Label_train_hsi = Data.TensorDataset(x_train,y_train)

    if is_train:
        x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
        # x_test_=torch.from_numpy(x_test_).type(torch.FloatTensor).permute(0,3,1,2) #[9671, 200, 7, 7]
        y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
        Label_test_hsi=Data.TensorDataset(x_test,y_test)
        # x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
        # y_true=torch.from_numpy(y_true).type(torch.LongTensor)
        # Label_true_hsi=Data.TensorDataset(x_true,y_true)

    # label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    # label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    # label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

    if is_train:
        return (Label_train_hsi, Label_test_hsi), (num_classes, band)
    else:
        return Label_train_hsi, (num_classes, band)


batch_size   = 192

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar100'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar100'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 35


networks = {}

pretrained = './experiments/CIFAR10_RotNet_ResNet18/model_net_epoch200'
# networks['feat_extractor'] = {'def_file': 'architectures/AlexNet.py', 'pretrained': pretrained, 'opt': {'num_classes': 4},  'optim_params': None} 
feat_net_opt = {'num_classes': 10, 'use_avg_on_conv3': False}
networks['feat_extractor'] = {'def_file': 'architectures/ResNet18.py', 'pretrained': pretrained, 'opt': feat_net_opt,  'optim_params': None} 

net_opt_cls = [None] * 5
# net_opt_cls = [None]
# net_opt_cls[0] = {'pool_type':'avg', 'nChannels':2048, 'pool_size':1, 'num_classes': 100}
net_opt_cls[0] = {'pool_type':'avg', 'nChannels':64, 'pool_size':8, 'num_classes': 100}
net_opt_cls[1] = {'pool_type':'avg', 'nChannels':256, 'pool_size':8, 'num_classes': 100}
net_opt_cls[2] = {'pool_type':'avg', 'nChannels':512, 'pool_size':4, 'num_classes': 100}
net_opt_cls[3] = {'pool_type':'avg', 'nChannels':1024, 'pool_size':2, 'num_classes': 100}
net_opt_cls[4] = {'pool_type':'avg', 'nChannels':2048, 'pool_size':1, 'num_classes': 100}
out_feat_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
# out_feat_keys = ['layer4']
net_optim_params_cls = {'optim_type': 'sgd', 'lr': 1e-3, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(5, 0.01),(15, 0.002),(25, 0.0004),(35, 0.00008)]}
networks['classifier']  = {'def_file': 'architectures/MultipleLinearClassifiers.py', 'pretrained': None, 'opt': net_opt_cls, 'optim_params': net_optim_params_cls}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'FeatureClassificationModel'
config['out_feat_keys'] = out_feat_keys


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import torchvision.models.resnet as resnet

from pdb import set_trace as breakpoint

class ResNet18(resnet.ResNet):
    def __init__(self, opt):
        super(ResNet18, self).__init__(resnet.Bottleneck, [2, 2, 2, 2], num_classes=opt['num_classes'])


    # def _parse_out_keys_arg(self, out_feat_keys):

    #     # By default return the features of the last layer / module.
    #     out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

    #     if len(out_feat_keys) == 0:
    #         raise ValueError('Empty list of output feature keys.')
    #     for f, key in enumerate(out_feat_keys):
    #         if key not in self.all_feat_names:
    #             raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
    #         elif key in out_feat_keys[:f]:
    #             raise ValueError('Duplicate output feature key: {0}.'.format(key))

    #     # Find the highest output feature in `out_feat_keys
    #     max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

    #     return out_feat_keys, max_out_feat
    

    def forward(self, x, out_feat_keys=None):
        # out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_feats.append(x)
        x = self.layer1(x)
        
        out_feats.append(x)
        x = self.layer2(x)
        
        out_feats.append(x)
        x = self.layer3(x)
        
        out_feats.append(x)
        x = self.layer4(x)

        out_feats.append(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x if out_feat_keys is None else out_feats

# class ResNet(resnet18):
#     def __init__(self, opt):
#         super(ResNet, self).__init__(pretrained=False)
#         num_classes = opt['num_classes']
        
#         self.resnet = resnet18(pretrained=False)
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, num_classes)
     
#     def _parse_out_keys_arg(self, out_feat_keys):

#         # By default return the features of the last layer / module.
#         out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

#         if len(out_feat_keys) == 0:
#             raise ValueError('Empty list of output feature keys.')
#         for f, key in enumerate(out_feat_keys):
#             if key not in self.all_feat_names:
#                 raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
#             elif key in out_feat_keys[:f]:
#                 raise ValueError('Duplicate output feature key: {0}.'.format(key))

#         # Find the highest output feature in `out_feat_keys
#         max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

#         return out_feat_keys, max_out_feat
    

def create_model(opt):
    # num_classes = opt['num_classes']
    
    # resnet = resnet18(pretrained=False)
    # num_ftrs = resnet.fc.in_features
    # resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    return ResNet18(opt)


if __name__ == '__main__':
    size = 224
    opt = {'num_classes':4}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

    out = net(x, out_feat_keys=net.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.all_feat_names[f], out[f].size()))

    filters = net.get_L1filters()

    print('First layer filter shape: {0}'.format(filters.size()))

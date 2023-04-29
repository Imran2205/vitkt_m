import torch.nn as nn
import torch
import torch.nn.functional as F
from tcopt import tcopts


class tclstm:
    def __init__(self,):
        '''
        in_dim: input feature dim (for bbox is 4)
        hidden_dim: output feature dim
        n_layer: num of hidden layers
        '''
        super(tclstm, self).__init__()
        self.conv_mapNet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, tcopts['map_units'], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))#mean
        )
        self.lstm1=nn.LSTM(10,64,1,batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, 1, batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 2))
        self._initialize_weights()
    def _initialize_weights(self):
        all_layers=[self.conv_mapNet,self.fc,self.lstm1,self.lstm2,self.lstm3]
        for layer in all_layers:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.normal_(param, 0, 0.01)
    def map_net(self, maps):
        map_outputs=self.conv_mapNet(maps)
        map_outputs=map_outputs.squeeze().unsqueeze(1).expand(-1,tcopts['time_steps'],-1)#[bs, tcopts['time_steps'], tcopts['map_units']]
        return map_outputs
    def net(self, inputs):
        outputs, states=self.lstm1(inputs)
        outputs,states=self.lstm2(outputs[:,-8:])
        outputs, states = self.lstm3(outputs[:,-3:])
        DQN_Inputs = outputs
        outputs=self.fc(outputs[:,-1])
        return outputs, DQN_Inputs

model=tclstm()
data=torch.rand(16,20,10)
output,dqn=model.net(data)
print(output.shape)

data=torch.rand(16,1,19,19)
output=model.map_net(data)
print(output.shape)


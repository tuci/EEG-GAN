import torch.nn as nn
import torch
import torch.nn.functional as F

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(manualSeed)

class Discriminator_DCGAN(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.in_dimen_class=50
        self.out_dimen_class=3750

        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
        )

        self.main = nn.Sequential(
            nn.Conv1d(2, 8, 64, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 8, 64, 3, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 8, 64, 3, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 8, 32, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 8, 16, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 8, 8, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 1, 4, bias=False),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)


    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            # nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        #print('class: {}'.format(y))
        out_input = x + torch.randn(x.shape).to('cuda')

        out_class = self.class_layer(y)
        out_class = out_class.view(-1, 1, self.out_dimen_class)

        out = torch.cat((out_input, out_class), dim=1)

        return self.main(out)

class Discriminator_WGAN_WC(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_WGAN_WC, self).__init__()
        self.ngpu = ngpu
        self.in_dimen = 456
        self.out_dimen = 1
        self.in_dimen_class = 50
        self.out_dimen_class = 3750
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
        )
        
        self.main = nn.Sequential(
            # convolution 1
            # input is (nc) x 3750
            nn.Conv1d(2, 8, 64, 2, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv1d(8, 8, 64, 2, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(8, 8, 64, 2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 64, 2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 64, 2, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.linear2 = nn.Linear(self.in_dimen, 1)
        # self.relu = nn.LeakyReLU(0.2)
    
    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            # nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        
        out_input = x + torch.randn(x.shape).to('cuda')
        
        out_class=self.class_layer(y)
        out_class=out_class.view(-1,1,self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.main(out)
        out = out.view(-1, self.in_dimen)
        out = self.linear2(out)
        return out

class Discriminator_Upsample_BigLinear_LayerNorm(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_Upsample_BigLinear_LayerNorm, self).__init__()
        self.ngpu = ngpu
        self.in_dimen = 456
        self.out_dimen = 1
        self.in_dimen_class = 50
        self.out_dimen_class = 3750
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
        )
        
        # self.main = nn.Sequential(
        # convolution 1
        # input is (nc) x 3750
        self.conv1 = nn.Conv1d(2, 8, 64, 2, bias=False)
        self.relu1 = nn.LeakyReLU(0.2)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv1d(8, 8, 64, 2, bias=False)
        # nn.BatchNorm1d(8),
        self.relu2 = nn.LeakyReLU(0.2)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv1d(8, 8, 64, 2, bias=False)
        # nn.BatchNorm1d(8),
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv1d(8, 8, 64, 2, bias=False)
        # nn.BatchNorm1d(8),
        self.relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv1d(8, 8, 64, 2, bias=False)
        # nn.BatchNorm1d(8),
        self.relu5 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(self.in_dimen, 1)
        self.relu = nn.LeakyReLU(0.2)

        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            # nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        
        out_input = x + torch.randn(x.shape).to('cuda')
        
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = F.layer_norm(out, out.size()[1:])
        out = self.relu2(out)
        out = self.conv3(out)
        out = F.layer_norm(out, out.size()[1:])
        out = self.relu3(out)
        out = self.conv4(out)
        out = F.layer_norm(out, out.size()[1:])
        out = self.relu4(out)
        out = self.conv5(out)
        out = F.layer_norm(out, out.size()[1:])
        out = self.relu5(out)
        out = out.view(-1, self.in_dimen)
        return self.relu(self.linear2(out))

class Discriminator_Upsample_BigLinear_BN_both(nn.Module):
    # Discriminator code
    def __init__(self, ngpu):
        super(Discriminator_Upsample_BigLinear_BN_both, self).__init__()
        self.ngpu = ngpu
        self.in_dimen = 456
        self.out_dimen = 1
        
        self.in_dimen_class = 50
        self.out_dimen_class = 3750
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
        )
        # self.main = nn.Sequential(
        # convolution 1
        # input is (nc) x 3750
        self.conv1 = nn.Conv1d(2, 8, 64, 2, bias=False)
        self.relu1 = nn.LeakyReLU(0.2)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv1d(8, 8, 64, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.LeakyReLU(0.2)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv1d(8, 8, 64, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv1d(8, 8, 64, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv1d(8, 8, 64, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(8)
        self.relu5 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(self.in_dimen, 1)
        # self.relu = nn.LeakyReLU(0.2)

        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            # nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        
        out_input = x + torch.randn(x.shape).to('cuda')
        
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = out.view(-1, self.in_dimen)
        return self.linear2(out)


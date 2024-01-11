import torch.nn as nn
import torch
import numpy as np

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(manualSeed)

# initialise bilinear deconvolution kernels
def init_bilinear_kernels(m):
    stride = m.stride[0]
    in_channels = m.in_channels
    out_channels = m.out_channels

    kernel_size = 2 * stride - stride % 2
    if kernel_size % 2 == 1:
        center = stride - 1
    else:
        center = stride - 0.5
    og = np.ogrid[:kernel_size]
    # define filter
    upsample_filter = torch.from_numpy(1 - np.abs(og - center) / stride)

    for in_channel in range(in_channels):
        for out_channel in range(out_channels):
            m.weight.data[in_channel,out_channel,:] = upsample_filter

# All generator classes
class Generator_WGAN_GP_Upsample_Deconv(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Upsample_Deconv, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 936
        self.in_dimen_class=50
        self.out_dimen_class=117
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
        )
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2)
        )
        self.up1 = nn.Upsample((234, 1), mode='bicubic')
        self.conv1 = nn.Sequential(
            nn.Conv1d(9, 8, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(     
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(     
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        # Output state
        self.conv2=nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x,nn.ConvTranspose1d):
            init_bilinear_kernels(x)
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        #out = self.main(input)
        #out = out.view(-1, 8, 117)
        # print(out)
        out = out.unsqueeze(dim=3)
        
        #Upsample layer 1
        out = self.up1(out).squeeze(dim=3)
        out = out.view(-1, 9, 234)
        out = self.conv1(out)
        
        out = self.deconv1(out) #deconvolutional layer 1
        out = self.deconv2(out) #deconvolutional layer 2
        out = self.deconv3(out) #deconvolutional layer 3
        out = self.deconv4(out) #deconvolutional layer 4
        out = self.conv2(out) #output
        return out

class Generator_WGAN_GP_Deconv_Normal(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Deconv_Normal, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 928
        self.in_dimen_class=50
        self.out_dimen_class=116

        self.class_layer = nn.Sequential(
                nn.Embedding(5,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(9, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.conv1 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self,x):
        if isinstance(x,nn.ConvTranspose1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x,nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x,nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.deconv1(out) #deconvolutional layer 1
        out = self.deconv2(out) #deconvolutional layer 2
        out = self.deconv3(out) #deconvolutional layer 3
        out = self.deconv4(out) #deconvolutional layer 4
        out = self.deconv5(out) #deconvolutional layer 5
        out = self.conv1(out) #output
        return out

class Generator_WGAN_GP_Deconv_Bilinear(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Deconv_Bilinear, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 928
        self.in_dimen_class=50
        self.out_dimen_class=116

        self.class_layer = nn.Sequential(
                nn.Embedding(5,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(9, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True))
        self.conv1 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x,nn.ConvTranspose1d):
            init_bilinear_kernels(x)
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)

    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.conv1(out)
        return out

class Generator_WGAN_GP_Upsample_BigLinear_BN_Gen(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Upsample_BigLinear_BN_Gen, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 1872
        self.in_dimen_class=50
        self.out_dimen_class=234
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))

        self.up1 = nn.Upsample((468, 1), mode='bicubic')
        self.conv1 = nn.Conv1d(9, 8, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.LeakyReLU(0.2)

        self.up2 = nn.Upsample((937 ,1), mode='bicubic')
        self.conv2 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.LeakyReLU(0.2)

        self.up3 = nn.Upsample((1875,1), mode='bicubic')
        self.conv3 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.LeakyReLU(0.2)

        self.up4 = nn.Upsample((3750,1), mode='bicubic')
        self.conv4 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
    
    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        # deconvolution layer 2
        out = out.unsqueeze(dim=3)
        out = self.up1(out).squeeze(dim=3)
        out = out.view(-1, 9, 468)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        # deconvolution layer 3
        out = out.view(-1, 8, 468)
        out = out.unsqueeze(dim=3)
        out = self.up2(out).squeeze(dim=3)
        out = out.view(-1, 8, 937)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # deconvolution layer 4
        out = out.view(-1, 8, 937)
        out = out.unsqueeze(dim=3)
        out = self.up3(out).squeeze(dim=3)
        out = out.view(-1, 8, 1875)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # deconvolution layer 5
        out = out.view(-1, 8, 1875)
        out = out.unsqueeze(dim=3)
        out = self.up4(out).squeeze(dim=3)
        out = out.view(-1, 8, 3750)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        return out

class Generator_WGAN_GP_Upsample_SmallLinear(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Upsample_SmallLinear, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 936
        self.in_dimen_class = 50
        self.out_dimen_class = 117

        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))

        self.up1 = nn.Upsample((234, 1), mode='bicubic')
        self.conv1 = nn.Conv1d(9, 8, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.LeakyReLU(0.2)

        self.up2 = nn.Upsample((468, 1), mode='bicubic')
        self.conv2 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.LeakyReLU(0.2)

        self.up3 = nn.Upsample((937 ,1), mode='bicubic')
        self.conv3 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.LeakyReLU(0.2)

        self.up4 = nn.Upsample((1875,1), mode='bicubic')
        self.conv4 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.LeakyReLU(0.2)

        self.up5 = nn.Upsample((3750,1), mode='bicubic')
        self.conv5 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(8)
        self.relu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
            
    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        # deconvolution layer 1
        out = out.unsqueeze(dim=3)
        out = self.up1(out).squeeze(dim=3)
        out = out.view(-1, 9, 234)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        # deconvolution layer 2
        out = out.view(-1, 8, 234)
        out = out.unsqueeze(dim=3)
        out = self.up2(out).squeeze(dim=3)
        out = out.view(-1, 8, 468)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # deconvolution layer 3
        out = out.view(-1, 8, 468)
        out = out.unsqueeze(dim=3)
        out = self.up3(out).squeeze(dim=3)
        out = out.view(-1, 8, 937)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # deconvolution layer 4
        out = out.view(-1, 8, 937)
        out = out.unsqueeze(dim=3)
        out = self.up4(out).squeeze(dim=3)
        out = out.view(-1, 8, 1875)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # deconvolution layer 5
        out = out.view(-1, 8, 1875)
        out = out.unsqueeze(dim=3)
        out = self.up5(out).squeeze(dim=3)
        out = out.view(-1, 8, 3750)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv6(out)

        return out

class Generator_WGAN_GP_Upsample_BigLinear_LayerNorm(nn.Module):
    def __init__(self):
        super(Generator_WGAN_GP_Upsample_BigLinear_LayerNorm, self).__init__()
        self.in_dimen_input=160
        self.out_dimen_input = 1872
        self.in_dimen_class=50
        self.out_dimen_class=234
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))
        
        self.up2 = nn.Upsample((468, 1), mode='bicubic')
        self.conv2 = nn.Conv1d(9, 8, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.LeakyReLU(0.2)

        self.up3 = nn.Upsample((937 ,1), mode='bicubic')
        self.conv3 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.relu3 = nn.LeakyReLU(0.2)

        self.up4 = nn.Upsample((1875,1), mode='bicubic')
        self.conv4 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.LeakyReLU(0.2)

        self.up5 = nn.Upsample((3750,1), mode='bicubic')
        self.conv5 = nn.Conv1d(8, 8, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(8)
        self.relu5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)

    def init_weights(self, x):
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
    
    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = out.unsqueeze(dim=3)
        out = self.up2(out).squeeze(dim=3)
        out = out.view(-1, 9, 468)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # deconvolution layer 3
        out = out.view(-1, 8, 468)
        out = out.unsqueeze(dim=3)
        out = self.up3(out).squeeze(dim=3)
        out = out.view(-1, 8, 937)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # deconvolution layer 4
        out = out.view(-1, 8, 937)
        out = out.unsqueeze(dim=3)
        out = self.up4(out).squeeze(dim=3)
        out = out.view(-1, 8, 1875)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # deconvolution layer 5
        out = out.view(-1, 8, 1875)
        out = out.unsqueeze(dim=3)
        out = self.up5(out).squeeze(dim=3)
        out = out.view(-1, 8, 3750)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        return out

class Generator_WGAN_WC(nn.Module):
    def __init__(self):
        super(Generator_WGAN_WC, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 936
        self.in_dimen_class=50
        self.out_dimen_class=117

        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )

        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(9, 8, 4, 2, 1,bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1,bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
            )
        self.deconv3 = nn.Sequential(
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv4 = nn.Sequential(
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Conv1d(8, 1, 1, bias=False)
        self.apply(self.init_weights)
    
    def init_weights(self, x):
        if isinstance(x, nn.ConvTranspose1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
    
    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.conv1(out)
        return out

class Generator_DCGAN(nn.Module):
    def __init__(self):
        super(Generator_DCGAN, self).__init__()
        self.in_dimen_input = 160
        self.out_dimen_input = 928
        self.in_dimen_class=50
        self.out_dimen_class=116
        
        self.class_layer = nn.Sequential(
                nn.Embedding(3,self.in_dimen_class),
                nn.Linear(self.in_dimen_class,self.out_dimen_class)
                )
        
        self.input_layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(self.in_dimen_input, self.out_dimen_input),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2))
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(9, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True))
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True))
        self.conv1 = nn.Sequential(
                nn.Conv1d(8, 1, 1, bias=False),
                nn.Tanh())
        self.apply(self.init_weights)
    
    def init_weights(self, x):
        if isinstance(x, nn.ConvTranspose1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.Conv1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
        if isinstance(x, nn.BatchNorm1d):
            nn.init.normal_(x.weight.data, 0.0, 0.02)
            nn.init.constant_(x.bias.data, 0)
    
    def forward(self, input):
        x, y = input
        # deconvolution layer 2
        out_class = self.class_layer(y)
        out_class = out_class.view(-1,1,self.out_dimen_class)
        
        out_input = self.input_layer(x)
        out_input = out_input.view(-1, 8, self.out_dimen_class)
        
        out = torch.cat((out_input,out_class),dim=1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.conv1(out)
        return out
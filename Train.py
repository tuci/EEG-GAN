import torch
import torch.nn.parallel

penalty_parameter = 10

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(manualSeed)

def train_dcgan(dataIterator, netD, netG, criterion, optimizerD, optimizerG,
                           device):
    # For each batch in the dataloader
    errD_avg = 0
    errG_avg = 0
    errD_G_avg = 0
    errD_fake_avg = 0
    errD_real_avg = 0
    total_instances_D = 0
    total_instances_G = 0
    D_losses = []
    G_losses = []

    real_label = 1
    fake_label = 0

    for i, data in enumerate(dataIterator, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        class_label = data[2].to(device)
        real_data = real_cpu[0, :, :]
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD([real_cpu, class_label]).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.empty((b_size, 1, 160))
        noise = torch.nn.init.normal_(noise).to(device)
        # Generate fake image batch with G
        fake = netG([noise, class_label])
        fake_data = fake[0, :, :]

        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD([fake.detach(), class_label]).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real - errD_fake
        # Update D
        optimizerD.step()

        errD_avg += errD.detach().item() * b_size
        errD_fake_avg += errD_fake.detach().item() * b_size
        errD_real_avg += errD_real.detach().item() * b_size
        total_instances_D += b_size

        # train critic 5 times more than generator
        trainD = 5
        if i % trainD == 0 and i != 0:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake = netG([noise, class_label])
            output = netD([fake, class_label]).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # scores
            errD_G_avg += errD.detach().item() * b_size
            errG_avg += errG.detach().item() * b_size
            total_instances_G += b_size

            G_losses.append(errG.item())
        # Output training stats
        if i % 50 == 0 and i != 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (i, len(dataIterator),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        D_losses.append(errD.item())

    scoresD = [errD_avg/total_instances_D, errD_fake_avg/total_instances_D, errD_real_avg/total_instances_D]
    scoresG = [errD_G_avg/total_instances_G, errG_avg/total_instances_G]
    return scoresD, scoresG, real_data, fake_data

def train_wgan_wc(dataIterator, netD, netG, optimizerD, optimizerG, device):
    # For each batch in the dataloader
    mean_err_D = 0
    mean_err_fake = 0
    mean_err_real = 0
    mean_err_G = 0
    mean_err_D_G = 0
    data_size = 0
    data_size_G = 0

    D_losses = []
    G_losses = []

    real_label = 1
    fake_label = 0
    for i, data in enumerate(dataIterator, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        class_label = data[2].to(device)
        real_data = real_cpu[0, :, :]
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD([real_cpu, class_label]).view(-1)
        # Calculate loss on all-real batch
        errD_real = output.mean(0).view(1)
        mean_err_real += errD_real.detach().item() * b_size
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.empty((b_size, 1, 160))
        noise = torch.nn.init.normal_(noise).to(device)
        # Generate fake image batch with G
        fake = netG([noise, class_label])
        fake_data = fake[0, :, :]

        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD([fake.detach(), class_label]).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = output.mean(0).view(1)
        mean_err_fake += errD_fake.detach().item() * b_size
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # calculate gradient penalty
        # gradient_penalty = calculate_gradient_penalty(netD, real_cpu, fake)
        # gradient_penalty.backward()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real - errD_fake
        # Update D
        optimizerD.step()
        data_size += b_size
        # clip weights
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        # train critic 5 times more than generator
        trainD = 5
        if i % trainD == 0 and i != 0:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake = netG([noise, class_label])
            output = netD([fake, class_label]).view(-1)
            # Calculate G's loss based on this output
            errG = - output.mean(0).view(1)
            mean_err_G += errG.detach().item() * b_size
            mean_err_D_G += errD.detach().item() * b_size
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = - errG
            # Update G
            optimizerG.step()
            data_size_G += b_size

            # Save Losses for plotting later
            G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Output training stats
        if i % 50 == 0 and i != 0:
            print('Batch: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataIterator),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    scoresD = [mean_err_D/data_size, mean_err_fake/data_size, mean_err_real/data_size]
    scoresG = [mean_err_D_G/data_size_G, mean_err_G/data_size_G]

    return scoresD, scoresG, real_data, fake_data

def train_wgan_gp(dataIterator, netD, netG, optimizerD, optimizerG, device):
    # For each batch in the dataloader
    mean_err_D = 0
    mean_err_fake = 0
    mean_err_real = 0
    mean_err_G = 0
    mean_err_D_G = 0
    mean_gp = 0
    data_size = 0
    data_size_G = 0

    D_losses = []
    G_losses = []

    real_label = 1
    fake_label = -1
    for i, data in enumerate(dataIterator, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        class_label = data[2].to(device)
        real_data = real_cpu[0, :, :]
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD([real_cpu, class_label]).view(-1)

        # Calculate loss on all-real batch
        errD_real = output.mean(0).view(1)
        mean_err_real += errD_real.detach().item() * b_size
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.empty((b_size, 1,160))
        noise = torch.nn.init.normal_(noise).to(device)
        # Generate fake image batch with G
        fake = netG([noise, class_label])
        fake_data = fake[0, :, :]

        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD([fake.detach(), class_label]).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = output.mean(0).view(1)
        mean_err_fake += errD_fake.detach().item() * b_size
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # calculate gradient penalty
        gradient_penalty = calculate_gradient_penalty(netD, real_cpu, fake, class_label)
        gradient_penalty.backward()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_fake - errD_real + gradient_penalty
        mean_err_D += errD.detach().item() * b_size
        # Update D
        optimizerD.step()
        mean_gp = (gradient_penalty.detach().item() / penalty_parameter) * b_size

        trainD = 5
        if i % trainD == 0 and i != 0:
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake = netG([noise, class_label])
            output = netD([fake, class_label]).view(-1)
            # Calculate G's loss based on this output
            errG = - output.mean(0).view(1)
            mean_err_G += errG.detach().item() * b_size
            mean_err_D_G += errD.detach().item() * b_size
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = errG
            # Update G
            optimizerG.step()
            data_size_G += b_size
            # Save Losses for plotting later
            G_losses.append(errG.item())
        D_losses.append(errD.item())
        data_size += b_size

        # Output training stats
        if i % 50 == 0 and i != 0:
            print('Batch: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tGP: %.4f'
                  % (i, len(dataIterator),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, gradient_penalty/penalty_parameter))

    scoresD = [mean_err_D/data_size, mean_err_fake/data_size, mean_err_real/data_size,
               mean_gp/data_size]
    scoresG = [mean_err_D_G/data_size_G, mean_err_G/data_size_G]
    return scoresD, scoresG, real_data, fake_data

def calculate_gradient_penalty(net, real, fake, class_label):
    # generate random variables
    bsize = real.shape[0]
    # generate random variables
    tens = torch.empty((bsize, 1, 1))
    t = torch.nn.init.uniform_(tens)
    t = t.to('cuda')

    interpolates = (t * real + ((1 - t) * fake)).to('cuda')
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    # forward pass
    output = net([interpolates, class_label])

    gradients = torch.autograd.grad(outputs=output, inputs=interpolates,
                                    grad_outputs=torch.ones(output.size()).to('cuda'))[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return torch.autograd.Variable(penalty_parameter * gradient_penalty, requires_grad=True)

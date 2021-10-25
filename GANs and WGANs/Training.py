import numpy as np
import torch
import yaml
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Training import Trainer
from GAN import gan_generator
from GAN import gan_discriminator
from WGAN import wgan_generator
from WGAN import wgan_discriminator
import torchvision
# from txt2image_dataset import Text2ImageDataset
# from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os

class Trainer(object):
    def __init__(self, type, dataset, lr = 0.0001, diter, vis_screen, save_path, l1_coef, l2_coef,\
                  batch_size = 64, num_workers = 2, epochs = 100):
        
        self.transform = transform = transforms.Compose([
                            transforms.ToTensor(), # convert to tensor
                            transforms.Lambda(lambda x: x.view(image_dim)) # flatten into vector
                            ])
        
        if type == "GAN":
            self.generator = torch.nn.DataParallel(gan_generator().cuda())
            self.discriminator = torch.nn.DataParallel(gan_discriminator().cuda())
        
        elif type == "WGAN":
            self.generator = torch.nn.DataParallel(wgan_generator().cuda())
            self.discriminator = torch.nn.DataParallel(wgan_discriminator().cuda())
        
        if dataset == 'clothes':
             self.dataset = torchvision.datasets.FashionMNIST(
                root='./data/FashionMNIST'
                ,train=True
                ,download=True
                ,transform=self.transform
                )

    
        elif dataset == 'numbers':
             self.dataset = torchvision.datasets.FashionMNIST(
                root='./data/MNIST'
                ,train=True
                ,download=True
                ,transform=self.transform
                )
        else:
            print('Dataset not supported, please select either clothes or numbers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,\
                                      shuffle=True, num_workers=self.num_workers)
 

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,\
                                       betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, \
                                       betas=(self.beta1, 0.999))

        self.logger = Logger(vis_screen)
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type

    def train(self, cls=False):
        if self.type == 'vanilla_wgan':
            self._train_vanilla_wgan()
        elif self.type == 'vanilla_gan':
            self._train_vanilla_gan()
    
    def _train_vanilla_wgan(self):
        one = Variable(torch.FloatTensor([1])).cuda()
        mone = one * -1
        gen_iteration = 0

        for epoch in range(self.num_epochs):
            iterator = 0
            data_iterator = iter(self.data_loader)

            while iterator < len(self.data_loader):
                
                # for the first 25 iterations the discriminator is trained a lot
                if gen_iteration < 25: 
                     d_iter_count = 100
                else:
                     d_iter_count = self.DITER

                d_iter = 0

                # Train the discriminator
                while d_iter < d_iter_count and iterator < len(self.data_loader):
                    d_iter += 1

                    for p in self.discriminator.parameters():
                         p.requires_grad = True

                    self.discriminator.zero_grad()

                    sample = next(data_iterator)
                    iterator += 1

                    right_images = sample['right_images']
                    right_images = Variable(right_images.float()).cuda()

                    outputs, _ = self.discriminator(right_images)
                    real_loss = torch.mean(outputs)
                    real_loss.backward(mone)

                    noise = Variable(torch.randn(right_images.size(0), \
                                                 self.noise_dim), volatile=True).cuda()
                    noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                    fake_images = Variable(self.generator(noise).data)
                    outputs, _ = self.discriminator(fake_images)
                    fake_loss = torch.mean(outputs)
                    fake_loss.backward(one)

                    ## NOTE: Pytorch had a bug with gradient penalty \
                    #at the time of this project development
                    ##       , uncomment the next two lines and remove the params 
                    # clamping below \
                    ##if you want to try gradient penalty
                    # gp = Utils.compute_GP(self.discriminator, right_images.data,\
                    #right_embed, fake_images.data, LAMBDA=10)
                    # gp.backward()

                    d_loss = real_loss - fake_loss
                    self.optimD.step()

                    for p in self.discriminator.parameters():
                         p.data.clamp_(-0.01, 0.01)

                # Train Generator
                for p in self.discriminator.parameters():
                     p.requires_grad = False

                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(noise)
                outputs, _ = self.discriminator(fake_images)

                g_loss = torch.mean(outputs)
                g_loss.backward(mone)
                g_loss = - g_loss
                self.optimG.step()

                gen_iteration += 1

                self.logger.draw(right_images, fake_images)
                self.logger.log_iteration_wgan(epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss)

        self.logger.plot_epoch(gen_iteration)

        if (epoch + 1) % 50 == 0:
             Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, epoch)

    def _train_vanilla_gan(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']

                right_images = Variable(right_images.float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()


                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(noise)
                outputs, _ = self.discriminator(fake_images)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(noise)
                outputs, activation_fake = self.discriminator(fake_images)
                _, activation_real = self.discriminator(right_images)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score)
                    self.logger.draw(right_images, fake_images)

            self.logger.plot_epoch_w_scores(iteration)

            if (epoch) % 50 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, epoch)

    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()

            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)
                
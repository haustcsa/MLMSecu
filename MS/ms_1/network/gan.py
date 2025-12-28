import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


def img_to_grayscale(img, freq):
    """Turn images to grayscale with frequency 1 / param"""
    if freq == 0:
        return img

    img_clone = img.clone()
    img_d = img_clone.detach()

    # Random channel selection
    rng_chan = torch.randint(3, (1,))[0]
    img_clone[::freq, (rng_chan + 1) % 3] = img_d[::freq, rng_chan]
    img_clone[::freq, (rng_chan + 2) % 3] = img_d[::freq, rng_chan]
    return img_clone


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def cond_print(s, t):
    if t:
        print(s)


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, activation=None, final_bn=True, grayscale=0):
        super(GeneratorA, self).__init__()
        self.grayscale = grayscale
        p = True
        
        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))
        cond_print("GeneratorA l1 size: {}".format(ngf*2*self.init_size**2), p)

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        cond_print("GeneratorA Conv2d size: {}".format(ngf*2), p)

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False) 
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False) 
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x :
            if self.grayscale != 0:
                raise NotImplementedError()
            return img
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            if self.grayscale != 0:
                return img_to_grayscale(self.activation(img), self.grayscale)
            return self.activation(img)


# DCGAN and Discriminator From Sanyal et. al. Towards Data-Free Model Stealing in a Hard Label Setting
class DCGAN(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64, grayscale=0):
        super(DCGAN, self).__init__()
        self.grayscale = grayscale
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input[:, :, None, None]

        output = self.main(input)
        if self.grayscale != 0:
            return img_to_grayscale(output, self.grayscale)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class DiffusionGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=32, activation=None, final_bn=True, grayscale=0, num_steps=1000):
        super(DiffusionGenerator, self).__init__()
        self.grayscale = grayscale
        self.num_steps = num_steps

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4  # Initial size after the first layer
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        # Conv block to reshape the output of the linear layer
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )

        # Fix the number of input channels for conv_blocks1
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final convolution block, optionally with batch normalization
        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                nn.BatchNorm2d(nc, affine=False),  # No affine parameters
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            )

    def forward(self, z, pre_x=False):
        """
        The forward pass for the generator. This generates an image from noise z.
        The generation process involves denoising (simulated diffusion reverse process).
        """
        # Step 1: Linear transformation from z to feature map
        out = self.l1(z.view(z.shape[0], -1))  # Flatten z and pass through first linear layer
        out = out.view(out.shape[0], -1, self.init_size,
                       self.init_size)  # Reshape to (batch_size, channels, height, width)

        # Step 2: Convolutional layers for image generation
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)  # Upsample by 2
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)  # Upsample by 2
        img = self.conv_blocks2(img)

        # Perform denoising (reverse diffusion process)
        # In a real diffusion model, this would involve multiple time steps, but we simulate it here
        denoised_img = self.denoise(img)

        if pre_x:
            if self.grayscale != 0:
                raise NotImplementedError("Grayscale not implemented for pre_x mode.")
            return denoised_img
        else:
            if self.grayscale != 0:
                return img_to_grayscale(self.activation(denoised_img), self.grayscale)
            return self.activation(denoised_img)

    def denoise(self, x_t, steps=1000):
        """
        A simplified denoising step that simulates the reverse diffusion process.
        In a real diffusion model, this would involve multiple time steps and more complex computations.
        Here we apply a simplified version in a single step.
        """
        # In practice, the denoising process would take multiple time steps, but for simplicity,
        # we'll simulate it with a single step here.

        # Typically, you'd need to apply the denoising step iteratively
        # Here we just pass the image through the generator's layers to simulate denoising
        for step in range(steps):
            x_t = self.denoising_step(x_t, step)

        return x_t

    def denoising_step(self, x_t, step):
        """
        A simplified denoising step for the image.
        In a real diffusion process, this would be more complex and depend on the timestep.
        Here we just apply a simple convolutional operation for denoising.
        """
        # Simulate the denoising process
        predicted_noise = self.predict_noise(x_t, step)
        return x_t - predicted_noise

    def predict_noise(self, x_t, step):
        """
        Predict the noise for the current timestep. In a real diffusion model, this would
        be learned, but here we'll simulate it with a convolution operation.
        """
        # Adjust the conv layer to handle the actual input size (3 channels) in the denoising process
        noise_pred = self.conv_blocks1(x_t)  # Just a placeholder for the noise prediction
        return noise_pred


def img_to_grayscale(img, grayscale_level):
    # Helper function to convert image to grayscale
    if grayscale_level == 1:
        return torch.mean(img, dim=1, keepdim=True)  # Convert to grayscale by averaging over channels
    return img  # No conversion if grayscale_level is 0


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from aw_nas.ops import register_primitive


# Operations of two dimensional UNet
# ---------------------------------------------------------------------------------------

class DownBlock2D(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int,
                 norm: str = "InstanceNorm2d", activation: str = "ReLU", inplace: bool = False):
        super(DownBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size = kernel_size, stride = stride, padding = padding)

        self.norm = norm
        if self.norm:
            self.norm1 = getattr(nn, norm)(C_out)
            self.norm2 = getattr(nn, norm)(C_out)

        try:
            self.act = getattr(nn, activation)(inplace = inplace)
        except:
            self.act = getattr(nn, activation)()

    def forward(self, x):
        output = self.conv1(x)
        if self.norm:
            output = self.norm1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.norm:
            output = self.norm2(output)
        output = self.act(output)
        return output


register_primitive("down_block_2d", 
                   lambda C, C_out, kernel_size = 3, stride = 1, padding = 1, norm = None, activation = "ReLU", inplace = False: \
                           DownBlock2D(C, C_out, kernel_size, stride, padding, norm, activation, inplace),
                    override = True
    )


class UpBlock2D(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int,
                 norm: str = "InstanceNorm2d", activation: str = "ReLU", inplace: bool = False):
        super(UpBlock2D, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode = "nearest")

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size = kernel_size, stride = stride, padding = padding)

        self.norm = norm
        if self.norm:
            self.norm1 = getattr(nn, norm)(C_out)
            self.norm2 = getattr(nn, norm)(C_out)

        try:
            self.act = getattr(nn, activation)(inplace = inplace)
        except:
            self.act = getattr(nn, activation)()

    def forward(self, x, y):
        output = torch.cat((self.upsample(x), y), dim = 1)
        output = self.conv1(output)
        if self.norm:
            output = self.norm1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.norm:
            output = self.norm2(output)
        output = self.act(output)
        return output


register_primitive("up_block_2d", 
        lambda C, C_out, kernel_size = 3, stride = 1, padding = 1, norm = None, activation = "ReLU", inplace = False: \
                UpBlock2D(C, C_out, kernel_size, stride, padding, norm, activation, inplace),
        override = True)

# End of operations of two dimensional UNet ---------------------------------------------


# Operations of three dimensional UNet
# ---------------------------------------------------------------------------------------

class DownBlock3D(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int,
                 batch_norm: bool = False, inplace: bool = False):
        super(DownBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(C_in, C_out // 2, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv3d(C_out // 2, C_out, kernel_size = kernel_size, stride = stride, padding = padding)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(C_out // 2)
            self.bn2 = nn.BatchNorm3d(C_out)

        self.relu = nn.ReLU(inplace = inplace)

    def forward(self, x):
        output = self.conv1(x)
        if self.batch_norm:
            output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        if self.batch_norm:
            output = self.bn2(output)
        output = self.relu(output)
        return output


register_primitive("down_block_3d", 
                   lambda C, C_out, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, inplace = False: \
                           DownBlock3D(C, C_out, kernel_size, stride, padding, batch_norm, inplace),
                   override = True)


class UpBlock3D(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int,
                 batch_norm: bool = False, inplace: bool = False):
        super(UpBlock3D, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode = "nearest")

        self.conv1 = nn.Conv3d(C_in, C_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv3d(C_out, C_out, kernel_size = kernel_size, stride = stride, padding = padding)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(C_out)
            self.bn2 = nn.BatchNorm3d(C_out)

        self.relu = nn.ReLU(inplace = inplace)

    def forward(self, x, y):
        output = torch.cat((self.upsample(x), y), dim = 1)
        output = self.conv1(output)
        if self.batch_norm:
            output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        if self.batch_norm:
            output = self.bn2(output)
        output = self.relu(output)
        return output


register_primitive("up_block_3d", 
                   lambda C, C_out, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, inplace = False: \
                           UpBlock3D(C, C_out, kernel_size, stride, padding, batch_norm, inplace),
                    override = True)

# End of operations of three dimensional UNet -------------------------------------------


# Operations of three dimensional Residual Net
# ---------------------------------------------------------------------------------------

class BasicBlock3D(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int = 1, 
                 downsample: bool = False, dilation: int = 1, 
                 padding: int = 1, batch_norm: bool = True) -> None:
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(C_in, C_out, kernel_size = 3, stride = stride, padding = padding, bias = False, dilation = dilation)
        self.relu = nn.ReLU(inplace = False)
        self.conv2 = nn.Conv3d(C_out, C_out, kernel_size = 3, stride = 1, padding = padding, bias = False, dilation = dilation)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm3d(C_out)
            self.bn2 = nn.BatchNorm3d(C_out)
        
        self.shortcut = nn.Sequential()
        
        assert (stride == 1 and C_in == C_out) or downsample, "'downsample' must be true if 'stride' > 1 or 'C_in != C_out'."

        if downsample:
            if self.batch_norm:
                self.shortcut = nn.Sequential(
                        nn.Conv3d(C_in, C_out, kernel_size = 1, stride = stride, bias = False),
                        nn.BatchNorm3d(C_out)
                        )
            else:
                self.shortcut = nn.Conv3d(C_in, C_out, kernel_size = 1, stride = stride, bias = False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


register_primitive("basic_block_3d", 
                   lambda C_in, C_out, stride = 1, downsample = False, padding = 1, dilation = 1, batch_norm = True: \
                           BasicBlock3D(C_in, C_out, stride, downsample, padding, dilation, batch_norm),
                    override = True)

# End of operations of three dimensional Residual Net -----------------------------------


# Operations of two dimensional discriminator
# ---------------------------------------------------------------------------------------

class VggBlock2D(nn.Module):
    def __init__(self, C: int, C_out: int, stride: int, affine: bool, inplace: bool = True,
                 norm: str = "InstanceNorm2d", activation: str = "ReLU"):
        super(VggBlock2D, self).__init__()
        self.conv = nn.Conv2d(C, C_out, kernel_size = 3, padding = 1)
        self.norm = getattr(nn, norm)(C_out) if norm else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) if stride == 2 else nn.Identity()
        self.act = getattr(nn, activation)(inplace = inplace)

    def forward(self, inputs):
        return self.pool(self.act(self.norm(self.conv(inputs))))


register_primitive("vgg_block_2d", 
                   lambda C, C_out, stride, affine = True, inplace = True, norm = "InstanceNorm2d", activation = "ReLU": 
                   VggBlock2D(C, C_out, stride, affine, inplace, norm, activation), 
                   override = True)

# End of operations of two dimensional discriminator ----------------------------------


# Operations of three dimensional discriminator
# ---------------------------------------------------------------------------------------

class VggBlock3D(nn.Module):
    def __init__(self, C: int, C_out: int, stride: int, affine: bool, inplace: bool = True,
                 norm: str = "InstanceNorm3d", activation: str = "ReLU"):
        super(VggBlock3D, self).__init__()
        self.conv = nn.Conv3d(C, C_out, kernel_size = 3, padding = 1)
        self.norm = getattr(nn, norm)(C_out) if norm else nn.Identity()
        self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2) if stride == 2 else nn.Identity()
        self.act = getattr(nn, activation)(inplace = inplace)

    def forward(self, inputs):
        return self.pool(self.act(self.norm(self.conv(inputs))))


register_primitive("vgg_block_3d", 
                   lambda C, C_out, stride, affine = True, inplace = True, norm = "InstanceNorm3d", activation = "ReLU": 
                   VggBlock3D(C, C_out, stride, affine, inplace, norm, activation), 
                   override = True)


class ConvELU3D(nn.Module):
    def __init__(self, C: int, C_out: int, kernel_size: int, 
                 stride: int, padding: int, inplace: bool = False):
        super(ConvELU3D, self).__init__()
        self.conv = nn.Conv3d(C, C_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.elu = nn.ELU(inplace)

    def forward(self, x):
        return self.elu(self.conv(x))


register_primitive("conv_elu_3d", 
                   lambda C, C_out, kernel_size, stride, padding, inplace = False: ConvELU3D(C, C_out, kernel_size, stride, padding, inplace),
                   override = True)


class ConvReLU3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ConvReLU3D, self).__init__()
        self.conv = nn.Conv3d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

register_primitive("conv_relu_3x3_3d", 
                   lambda C, C_out, stride, affine: ConvReLU3D(C, C_out, 3, stride, 1),
                   override = True)

register_primitive("conv_relu_3x3_3d_c1", 
                   lambda C, C_out, stride, affine: ConvReLU3D(1, C_out, 3, stride, 1),
                   override = True)

# End of operations of three dimensional discriminator ----------------------------------
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    inputs = torch.ones(2, 16, 40, 200, 200)
    net = BasicBlock3D(C_in = 16, C_out = 32, stride = 2, downsample = True, padding = 2, dilation = 2, batch_norm = False)
    outputs = net(inputs)
    loss = outputs.sum()
    loss.backward()
    print(outputs.shape)

import torch
import torch.nn as nn

from aw_nas import ops
from aw_nas.final.base import FinalModel


# 2D network
# --------------------------------------------------------------------------------

class TwoDimensionUNet(FinalModel):
    """
    2D-UNET for image restoration.
    
    Args:
        * C_in: [int] channel number of the input
        * C_out: [int] channel number of the output
        * init_channel: [int] initial channel number of the network
        * depth: [int] depth of the network
        * norm: [str] type of normalization layer (choice: ["InstanceNorm2d", "BatchNorm2d", None])
        * activation: [str] type of the activation function
        * inplace: [bool]
    """
    
    NAME = "two_dimension_unet"

    def __init__(self, device, C_in: int = 1, C_out: int = 1, init_channel: int = 64, depth: int = 5, 
                 norm: str = "InstanceNorm2d", activation: str = "ReLU", inplace: bool = False,
                 search_space = None, schedule_cfg = None):
        super(TwoDimensionUNet, self).__init__(schedule_cfg)
        self.depth = depth
        self.encoder_list = nn.ModuleList([
            ops.get_op("down_block_2d")(
                C = C_in if i == 0 else 2 ** (i - 1) * init_channel, 
                C_out = 2 ** i * init_channel, 
                kernel_size = 3,
                stride = 1,
                padding = 1,
                norm = norm,
                activation = activation,
                inplace = inplace
                ) \
                        for i in range(depth)
            ])
        self.decoder_list = nn.ModuleList([
            ops.get_op("up_block_2d")(
                C = 2 ** i * init_channel * 3,
                C_out = 2 ** i * init_channel, 
                inplace = inplace, 
                norm = norm,
                activation = activation
                ) \
                        for i in range(depth - 1)
            ])
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.conv = nn.Conv2d(init_channel, C_out, kernel_size = 3, stride = 1, padding = 1)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        down_states = []
        for i, encoder in enumerate(self.encoder_list):
            down_states.append(encoder(x if i == 0 else self.maxpool(down_states[-1])))
        output = down_states[-1]
        for i, decoder in enumerate(reversed(self.decoder_list)):
            output = decoder(output, down_states[-(i + 2)])
        output = self.conv(output)
        return output
    
    def supported_data_types(self):
       return ["image"]


class TwoDimensionDiscriminator(FinalModel):
    """
    Two-dimensional discriminator.
    
    Args:
        * C_in: [int] channel number of the input
        * num_classes: [int] output class number
        * init_channel: [int] initial channel number of the network
        * linear_dim: [int] dimension of the final linear layer
        * affine: [bool]
        * inplace: [bool]
        * norm: [str] type of normalization layer (choice: ["InstanceNorm2d", "BatchNorm2d", None])
        * activation: [str] type of the activation function
    """

    NAME = "two_dimension_discriminator"

    def __init__(self, device, C_in: int, num_classes: int = 2, init_channel: int = 16, linear_dim = 6400,
                 affine: bool = True, inplace: bool = False, norm: str = "InstanceNorm2d", activation: str = "ReLU", search_space = None, schedule_cfg = None):
        super(TwoDimensionDiscriminator, self).__init__(schedule_cfg)
        self.stem = nn.Sequential(
                nn.Conv2d(C_in, init_channel, kernel_size = 3, padding = 1),
                getattr(nn, norm)(init_channel) if norm else nn.Identity()
                )

        self.vgg_blocks = nn.Sequential(*[
            ops.get_op("vgg_block_2d")(init_channel, 2 * init_channel, 1, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(2 * init_channel, 4 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(4 * init_channel, 8 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(8 * init_channel, 8 * init_channel, 1, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(8 * init_channel, 8 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(8 * init_channel, 4 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(4 * init_channel, 2 * init_channel, 1, affine, inplace, norm, activation),
            ops.get_op("vgg_block_2d")(2 * init_channel, init_channel, 1, affine, inplace, norm, activation),
            ])

        self.linear = nn.Linear(linear_dim, num_classes)
        
        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.vgg_blocks(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.linear(outputs)
    
    def supported_data_types(self):
       return ["image"]

# End of 2D network -----------------------------------------------------------
# -----------------------------------------------------------------------------


# 3D network
# -----------------------------------------------------------------------------

class ThreeDimensionTwoDimensionBasedUNet(TwoDimensionUNet):
    """
    Two-dimensional-UNET based three-dimensional network.
    Reduce the metal artifact in each slice with metal artifact selected from 3D data with our proposed "Threshold discriminant method"
    
    Args:
        * threshold: [float] 
    """

    NAME = "three_dimension_two_based_unet"
    
    def __init__(self, threshold: float = 2.0, **kwargs):
        super(ThreeDimensionTwoDimensionBasedUNet, self).__init__(**kwargs)
        self.threshold = threshold

    def forward(self, x):
        # Calculate the attention map
        origin_input = torch.clone(x)
        artifact_map = torch.sum(torch.sum(origin_input > self.threshold, dim = -1), dim = -1) > 0
        # Generate the new batch
        # x: [batch size, channel, depth, length, width] -> 
        # input: [batch size, channel, length, width]
        input = origin_input[artifact_map].unsqueeze(dim = 1)
        if len(input) == 0:
            return origin_input
        # Inner forward
        down_states = []
        for i, encoder in enumerate(self.encoder_list):
            down_states.append(encoder(input if i == 0 else self.maxpool(down_states[-1])))
        output = down_states[-1]
        for i, decoder in enumerate(reversed(self.decoder_list)):
            output = decoder(output, down_states[-(i + 2)])    
        output = self.conv(output)
        # Construct the final output
        origin_input[artifact_map] = output.squeeze(dim = 1)
        return origin_input


class ThreeDimensionResidualNet(FinalModel):
    """
    Three-dimensional ResidualNet.
    Reduce the metal artifact in each slice with metal artifact selected from 3D data with our proposed "Threshold discriminant method".
    
    Args:
        * threshold: [float] 
        * C_in: [int] input channel number
        * init_channel: [int] initial channel number of the network
        * batch_norm: [bool] whether use BN in stem
        * genotype: [list]
    """
    
    NAME = "three_dimension_resnet"

    def __init__(self, device, C_in: int, init_channel: int, batch_norm: bool,
                 genotype: list, search_space = None, schedule_cfg = None):
        super(ThreeDimensionResidualNet, self).__init__(schedule_cfg)
        if batch_norm:
            self.stem = nn.Sequential(
                nn.Conv3d(C_in, init_channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm3d(init_channel)
                )
        else:
            self.stem = nn.Conv3d(C_in, init_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.blocks = nn.ModuleList([ops.get_op(operation["operation_type"])(**operation["operation_kwargs"]) for operation in genotype])
        self.device = device
        self.to(self.device)

    def forward(self, x):
        output = self.stem(x)
        for block in self.blocks:
            output = block(output)
        return output

    def supported_data_types(self):
        return ["image"]


class ThreeDimensionUNet(FinalModel):
    """
    3D-UNET for image restoration.
    
    Args:
        * C_in: [int] channel number of the input
        * C_out: [int] channel number of the output
        * init_channel: [int] initial channel number of the network
        * depth: [int] depth of the network
        * batch_norm: whether use BN
        * inplace: [bool]
    """
    
    NAME = "three_dimension_unet"

    def __init__(self, device, 
                 C_in, C_out, init_channel: int = 32, depth: int = 4, 
                 batch_norm: bool = False, inplace: bool = False,
                 search_space = None, schedule_cfg = None):
        super(ThreeDimensionUNet, self).__init__(schedule_cfg)
        self.depth = depth
        self.encoder_list = nn.ModuleList([
            ops.get_op("down_block_3d")(
                C = C_in if i == 0 else 2 ** i * init_channel, 
                C_out = 2 ** (i + 1) * init_channel, 
                kernel_size = 3,
                stride = 1,
                padding = 1,
                batch_norm = batch_norm,
                inplace = inplace
                ) \
                        for i in range(depth)
            ])
        self.decoder_list = nn.ModuleList([
            ops.get_op("up_block_3d")(
                C = 2 ** i * init_channel * 3,
                C_out = 2 ** i * init_channel, 
                inplace = inplace, 
                batch_norm = batch_norm
                ) \
                        for i in range(1, depth)
            ])
        self.maxpool = nn.MaxPool3d(2, stride = 2)
        self.conv = nn.Conv3d(2 * init_channel, C_out, kernel_size = 3, stride = 1, padding = 1)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        down_states = []
        for i, encoder in enumerate(self.encoder_list):
            down_states.append(encoder(x if i == 0 else self.maxpool(down_states[-1])))
        output = down_states[-1]
        for i, decoder in enumerate(reversed(self.decoder_list)):
            output = decoder(output, down_states[-(i + 2)])
        output = self.conv(output)
        return output
    
    def supported_data_types(self):
       return ["image"]


class ThreeDimensionDiscriminator(TwoDimensionDiscriminator):
    """
    Three-dimensional discriminator.
    
    Args:
        * C_in: [int] channel number of the input
        * num_classes: [int] output class number
        * init_channel: [int] initial channel number of the network
        * linear_dim: [int] dimension of the final linear layer
        * affine: [bool]
        * inplace: [bool]
        * norm: [str] type of normalization layer (choice: ["InstanceNorm3d", "BatchNorm3d", None])
        * activation: [str] type of the activation function
    """

    NAME = "three_dimension_discriminator"
    
    def __init__(self, device, C_in: int, num_classes: int, init_channel: int, 
                 linear_dim: int, affine: bool = True, inplace: bool = False, 
                 norm: str = "InstanceNorm3d", activation: str = "ReLU", 
                 search_space = None, schedule_cfg = None):
        FinalModel.__init__(self, schedule_cfg)

        self.stem = nn.Sequential(
                nn.Conv3d(C_in, init_channel, kernel_size = 3, padding = 1),
                getattr(nn, norm)(init_channel) if norm else nn.Identity()
                )

        self.vgg_blocks = nn.Sequential(*[
            ops.get_op("vgg_block_3d")(init_channel, 2 * init_channel, 1, affine, inplace, norm, activation),
            ops.get_op("vgg_block_3d")(2 * init_channel, 4 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_3d")(4 * init_channel, 8 * init_channel, 2, affine, inplace, norm, activation),
            ops.get_op("vgg_block_3d")(8 * init_channel, 16 * init_channel, 2, affine, inplace, norm, activation)
            ])

        self.linear = nn.Linear(linear_dim, num_classes)
        
        self.device = device
        self.to(self.device)


class ThreeDimensionDiscriminatorV2(FinalModel):
    """
    Another three-dimensional discriminator.
    
    Args:
        * C_in: [int] channel number of the input
        * num_classes: [int] output class number
        * init_channel: [int] initial channel number of the network
    """

    NAME = "three_dimension_discriminator_v2"

    def __init__(self, device, C_in: int, num_classes: int = 2, init_channel: int = 16, search_space = None, schedule_cfg = None):
        super(ThreeDimensionDiscriminatorV2, self).__init__(schedule_cfg)

        self.ops = nn.Sequential(
                ops.get_op("conv_elu_3d")(C_in, init_channel, 3, 1, 1, False),
                nn.Conv3d(init_channel, init_channel, kernel_size = 1, stride = 1, padding = 0),
                nn.AvgPool3d(kernel_size = 2, stride = 2),
                ops.get_op("conv_elu_3d")(init_channel, init_channel // 2, 3, 1, 1, False),
                nn.Conv3d(init_channel // 2 , init_channel // 2, kernel_size = 1, stride = 1, padding = 0),
                nn.AvgPool3d(kernel_size = 2, stride = 2),
                ops.get_op("conv_elu_3d")(init_channel // 2, 1, 3, 1, 1, False)
                )
        self.linear = nn.Linear(84375, num_classes)
        
        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        outputs = self.ops(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.linear(outputs)
    
    def supported_data_types(self):
       return ["image"]

# End of 3D network -----------------------------------------------------------
# -----------------------------------------------------------------------------


def test_three_dimension_unet():
    from aw_nas.utils import RegistryMeta
    cfg = {
            "final_model_type": "three_dimension_unet",
            "final_model_cfg": {
                "C_in": 1,
                "C_out": 1,
                "device": None,
                "init_channel": 10,
                "depth": 4,
                "batch_norm": False,
                "inplace": False
                }
            }
    type_ = cfg["final_model" + "_type"]
    cfg = cfg.get("final_model" + "_cfg", None)
    cls = RegistryMeta.get_class("final_model", type_)
    net = cls(**cfg).cuda()

    inputs = torch.ones((1, 1, 64, 320, 320)).cuda()
    print(net(inputs).shape)


def test_two_dimension_unet():
    from aw_nas.utils import RegistryMeta
    cfg = {
            "final_model_type": "two_dimension_unet",
            "final_model_cfg": {
                "C_in": 1,
                "C_out": 1,
                "device": None,
                "init_channel": 64,
                "depth": 5,
                "norm": "InstanceNorm2d",
                "activation": "ReLU",
                "inplace": False
                }
            }
    type_ = cfg["final_model" + "_type"]
    cfg = cfg.get("final_model" + "_cfg", None)
    cls = RegistryMeta.get_class("final_model", type_)
    net = cls(**cfg).cuda()

    inputs = torch.ones((1, 1, 320, 320)).cuda()
    print(net(inputs).shape)


def test_two_dimensional_discriminator():
    from aw_nas.utils import RegistryMeta
    cfg = {
            "final_model_type": "two_dimension_discriminator",
            "final_model_cfg": {
                "C_in": 1,
                "num_classes": 2,
                "device": None,
                "init_channel": 10,
                "norm": "InstanceNorm2d",
                "affine": True,
                "inplace": True, 
                "activation": "ReLU",
                "linear_dim": 4000
                }
            }
    type_ = cfg["final_model" + "_type"]
    cfg = cfg.get("final_model" + "_cfg", None)
    cls = RegistryMeta.get_class("final_model", type_)
    net = cls(**cfg).cuda()

    inputs = torch.ones((21, 1, 320, 320)).cuda()
    print(net(inputs).shape)
 

def test_three_dimensional_discriminator():
    from aw_nas.utils import RegistryMeta
    cfg = {
            "final_model_type": "three_dimension_discriminator",
            "final_model_cfg": {
                "C_in": 1,
                "num_classes": 2,
                "device": None,
                "init_channel": 16,
                "norm": "InstanceNorm3d",
                "affine": True,
                "inplace": True, 
                "activation": "ReLU",
                "linear_dim": 6400
                }
            }
    type_ = cfg["final_model" + "_type"]
    cfg = cfg.get("final_model" + "_cfg", None)
    cls = RegistryMeta.get_class("final_model", type_)
    net = cls(**cfg).cuda()

    inputs = torch.ones((1, 1, 64, 320, 320)).cuda()
    print(net(inputs).shape)


if __name__ == "__main__":
    test_two_dimensional_discriminator()

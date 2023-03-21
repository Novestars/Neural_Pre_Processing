import nibabel
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from models.losses import SSIM3D
from einops import rearrange
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def __call__(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        grad = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

        return grad


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,stride = 1,relu='lrelu'):
        super().__init__()
        if relu=='lrelu':
            self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1,stride=stride)
        self.bn1 = nn.InstanceNorm3d(middle_channels,affine=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels,affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1,):
        super().__init__()
        self._leaky_relu_alpha = 0.01
        nb_filter = [16, 32, 64, 128, 256,512]
        self.nb_filter = nb_filter
        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],stride=2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],stride=2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],stride=2)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],stride=2)
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5],stride=2)

        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], output_channels, kernel_size=1)
        self.LN1 = nn.LayerNorm(nb_filter[5])
        self.LN2 = nn.LayerNorm(nb_filter[5])
        self.attention1 = Attention(nb_filter[5],heads = 8,dim_head=64)
        self.mlp1 = FeedForward(nb_filter[5],nb_filter[5])
        self.LN3 = nn.LayerNorm(nb_filter[5])
        self.LN4 = nn.LayerNorm(nb_filter[5])
        self.attention2 = Attention(nb_filter[5],heads = 8,dim_head=64)
        self.mlp2 = FeedForward(nb_filter[5],nb_filter[5])

        self.head = nn.Sequential(
            nn.Linear(nb_filter[5], nb_filter[5] // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nb_filter[5] // 2, 12, bias=False),
            nn.Tanh()
        )
        self.head2 = nn.Sequential(
            nn.Linear(1, nb_filter[5]),
            nn.ReLU(inplace=True),
            nn.Linear(nb_filter[5], nb_filter[5]*4),
            nn.ReLU(inplace=True),
            nn.Linear(nb_filter[5]*4, sum(np.array(nb_filter[:-1]))*2),
        )
    def forward(self, input,weight=None):

        input_downsampled = torch.nn.functional.interpolate(input, size=[128, 128, 128],
                                                mode='trilinear', align_corners=False)

        x0_0 = self.conv0_0(input_downsampled)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)

        identity = torch.eye(3,4).repeat(x5_0.shape[0],1,1).type_as(x5_0)
        b,c,h,w,d = x5_0.shape
        x5_0_faltten = rearrange(x5_0,'b c h w d-> b (h w d) c')
        x5_0_faltten = self.attention1(self.LN1(x5_0_faltten)) + x5_0_faltten
        x5_0_faltten = self.mlp1(self.LN2(x5_0_faltten)) + x5_0_faltten
        x5_0_faltten = self.attention2(self.LN3(x5_0_faltten)) + x5_0_faltten
        x5_0_faltten = self.mlp2(self.LN4(x5_0_faltten)) + x5_0_faltten

        affine = self.head(x5_0_faltten.mean(dim=1)).reshape(-1, 3, 4) + identity
        x5_0 = rearrange(x5_0_faltten,'b (h w d) c -> b c h w d',h=h,w=w,d=d)

        x0_0_warp = torch.nn.functional.affine_grid(affine, input.size(), align_corners=False)
        mod = self.head2(torch.Tensor([weight]).type_as(x0_0_warp))
        mod = torch.split(mod,np.repeat(self.nb_filter[:-1],2).tolist(),0)

        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))*mod[-1].reshape(1,-1,1,1,1)+ mod[-2].reshape(1,-1,1,1,1)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))*mod[-3].reshape(1,-1,1,1,1) + mod[-4].reshape(1,-1,1,1,1)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))*mod[-5].reshape(1,-1,1,1,1)+ mod[-6].reshape(1,-1,1,1,1)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))*mod[-7].reshape(1,-1,1,1,1)+ mod[-8].reshape(1,-1,1,1,1)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))*mod[-9].reshape(1,-1,1,1,1)+ mod[-10].reshape(1,-1,1,1,1)
 
        output = self.final(x0_1)
        output_upsampled = torch.nn.functional.interpolate(output, size=[256, 256, 256],mode='trilinear', align_corners=False)
        norm = input*255 * output_upsampled
        mni_norm = torch.nn.functional.grid_sample(norm, x0_0_warp, align_corners=False)

        return mni_norm,norm,output_upsampled
class NPP(LightningModule): 
    def __init__(
        self,
        lr,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = UNet()

        self.ssim7 = SSIM3D(window_size=5)
        self.mse = nn.L1Loss()
        self.grad = Grad('l1')
        self.automatic_optimization = False


    def forward(self, x,ind = None):
        return self.generator(x,ind)

    def training_step(self, batch, batch_idx):
        imgs, gts,brain_mask, ind = batch
        opt = self.optimizers()
        sch = self.lr_schedulers()
        # train generator
        # generate images
        weight = np.random.uniform(2,-4)
        self.generated_imgs = self(imgs,weight)

        # log sampled images
        loss_ssim_7 = -self.ssim7(self.generated_imgs[1], gts)

        grad_loss = self.grad(self.generated_imgs[0])

        loss = loss_ssim_7 + grad_loss*10**weight
        opt.zero_grad()

        self.manual_backward(loss)
        opt.step()

        if self.trainer.is_last_batch:
            sch.step()

        self.log("tv_train" , grad_loss,prog_bar=True, sync_dist=True)
        self.log("ssim_7_train" , loss_ssim_7,prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        my_list = []
        sparse_params = list(filter(lambda kv: kv[0] in my_list, self.generator.named_parameters()))
        sparse_params = [i[1] for i in sparse_params]
        base_params = list(filter(lambda kv: kv[0] not in my_list, self.generator.named_parameters()))
        base_params = [i[1] for i in base_params]

        base_params = [{"params": base_params},]
        sparse_params = [{"params": sparse_params},]

        optimizer = torch.optim.Adam(base_params, lr=lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.5)
        return [optimizer], [lr_scheduler,]

    def validation_step(self, batch, batch_idx):
        imgs, gts,brain_mask,ind = batch

        # train generator
        self.generated_imgs = self(imgs,0.1)

        grad_loss = self.grad(self.generated_imgs[0])
        loss_ssim_7 = -self.ssim7(self.generated_imgs[1],gts)
        loss = grad_loss + loss_ssim_7
        self.log("tv_val", grad_loss,prog_bar=True, sync_dist=True )
        self.log("ssim_7_val", loss_ssim_7,prog_bar=True, sync_dist=True )

        return loss








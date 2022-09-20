from .tensor_base import TensorVoxelBase
import torch
import torch.nn.functional as F
from icecream import ic
from models.grid_sample_Cinf import grid_sample
import random
import math
from models import safemath

# here is original grid sample derivative for testing
# def grid_sample(*args, smoothing, **kwargs):
#     return F.grid_sample(*args, **kwargs)

def d_softplus(x, beta=1.0, shift=-10):
    return torch.exp(shift+beta*x) / (1.0 + torch.exp(shift+beta*x))


class TensorVMSplit(TensorVoxelBase):
    def __init__(self, aabb, init_mode='trig', *args, smoothing, **kargs):
        super(TensorVMSplit, self).__init__(aabb, *args, **kargs)

        # num_levels x num_outputs
        self.interp_mode = 'bilinear'
        self.init_mode = init_mode
        # self.interp_mode = 'bicubic'
        self.align_corners = True

        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, [int(self.density_res_multi*g) for g in self.grid_size], 0.1, -0)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.grid_size, 0.1, 0)
        m = sum(self.app_n_comp)
        self.basis_mat = torch.nn.Linear(m, self.app_dim, bias=False)
        self.dbasis_mat = torch.nn.Linear(sum(self.density_n_comp), 1, bias=False)

        self.smoothing = smoothing

    def init_one_svd(self, n_component, grid_size, scale, shift):
        plane_coef, line_coef = [], []

        xyg = torch.meshgrid(torch.linspace(-1, 1, grid_size[0]), torch.linspace(-1, 1, grid_size[1]), indexing='ij')
        xy = xyg[0] + xyg[1]

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            pos_vals = xy.reshape(1, 1, grid_size[mat_id_0], grid_size[mat_id_1])
            # freqs = torch.arange(n_component[i]//2).reshape(1, -1, 1, 1)
            n_degs = n_component[i]//2
            freqs = 2**torch.arange(n_degs-1).reshape(1, -1, 1, 1)
            freqs = torch.cat([torch.zeros_like(freqs[:, 0:1]), freqs], dim=1)
            line_pos_vals = torch.linspace(-1, 1, grid_size[vec_id]).reshape(1, 1, -1, 1)
            scales = scale * 1/(freqs+1)
            # scales[:, scales.shape[1]//2:] = 0
            match self.init_mode:
                case 'trig':
                    plane_coef_v = torch.cat([
                        scales * torch.sin(freqs * pos_vals * math.pi),
                        scales * torch.cos(freqs * pos_vals * math.pi),
                    ], dim=1)
                    line_coef_v = torch.cat([
                        scales * torch.sin(freqs * line_pos_vals * math.pi),
                        scales * torch.cos(freqs * line_pos_vals * math.pi),
                    ], dim=1)
                case 'integrated':
                    b = safemath.integrated_pos_enc((pos_vals.reshape(-1, 1)*torch.pi, torch.ones_like(pos_vals).reshape(-1, 1)), 0, n_degs)
                    b = b.T.reshape(1, b.shape[1], *pos_vals.shape[-2:])

                    a = safemath.integrated_pos_enc((line_pos_vals.reshape(-1, 1)*torch.pi, torch.ones_like(line_pos_vals).reshape(-1, 1)), 0, n_degs)
                    a = a.T.reshape(1, a.shape[1], *line_pos_vals.shape[-2:])
                    plane_coef_v = b
                    line_coef_v = a
                case 'rand':
                    plane_coef_v = scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))
                    line_coef_v = scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))
            plane_coef.append(torch.nn.Parameter(plane_coef_v))
            line_coef.append(torch.nn.Parameter(line_coef_v))

        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    
    
    def get_optparam_groups(self):
        grad_vars = [
            {'params': self.density_line, 'lr': self.lr}, {'params': self.density_plane, 'lr': self.lr},
            {'params': self.app_line, 'lr': self.lr}, {'params': self.app_plane, 'lr': self.lr},
            {'params': self.basis_mat.parameters(), 'lr': self.lr_net},
            {'params': self.dbasis_mat.parameters(), 'lr': self.lr_net},
        ]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg, start_ind=0, end_ind=-1):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total

    def coordinates(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def compute_densityfeature(self, xyz_sampled):
        # mask1 = torch.linalg.norm(xyz_sampled[..., :3], dim=-1, ord=torch.inf) < 0.613/1.5
        # mask2 = (xyz_sampled[..., 0] < 0) & (xyz_sampled[..., 1] > 0)
        # return torch.where(mask1 & ~mask2, 99999999.0, 0.0)

        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        sigma_feature = []

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_sampled.shape[:1])
            sigma_feature.append(plane_coef_point * line_coef_point)

        # return self.dbasis_mat(sigma_feature.reshape(-1, 1)).reshape(-1)
        sigma_feature = torch.cat(sigma_feature, dim=0).T
        # ic(sigma_feature[0], sigma_feature[0].sum())
        sigma_feature = self.dbasis_mat(sigma_feature).squeeze(-1)
        # ic(list(self.dbasis_mat.parameters()))
        # sigma_feature = (sigma_feature).sum(dim=1).squeeze(-1)
        # sigma_feature = sigma_feature.sum(dim=-1)
        return self.feature2density(sigma_feature)


    def compute_appfeature(self, xyz_sampled):
        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
        plane_coef_point,line_coef_point = [],[]
        # plane_kerns = [self.norm_plane_kernels[0][0:1]]
        # line_kerns = [self.norm_line_kernels[0][0:1]]
        plane_kerns, line_kerns = [[None]], [[None]]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                    F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interp_mode,
                        align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(
                    F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]], mode=self.interp_mode,
                        align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point, dim=0), torch.cat(line_coef_point, dim=0)
        return self.basis_mat((plane_coef_point * line_coef_point).T)


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode=self.interp_mode,
                              align_corners=self.align_corners))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode=self.interp_mode, align_corners=self.align_corners))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        density_target = [int(self.density_res_multi*g) for g in res_target]
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, density_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {density_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        # the new_aabb is in normalized coordinates, from -1 to 1
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        # t_l, b_r = xyz_min * self.grid_size // 2, xyz_max * self.grid_size // 2 - 1
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        dt_l, db_r = torch.floor(t_l*self.density_res_multi).long(), torch.ceil(b_r*self.density_res_multi).long() + 1
        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        db_r = torch.stack([db_r, (self.density_res_multi*self.grid_size).long()]).amin(0)

        # update aabb
        l1 = t_l / self.grid_size
        l2 = b_r / self.grid_size
        adj_aabb = torch.stack([
            l1 * self.aabb[1] + (1-l1) * self.aabb[0],
            l2 * self.aabb[1] + (1-l2) * self.aabb[0],
        ], dim=0)
        ic(db_r, dt_l, b_r, t_l, xyz_min, xyz_max, self.units, self.aabb, adj_aabb, self.density_line[0].shape, self.grid_size)
        self.aabb = adj_aabb

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,dt_l[mode0]:db_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,dt_l[mode1]:db_r[mode1],dt_l[mode0]:db_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        newSize = b_r - t_l
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorVoxelBase):
    def __init__(self, aabb, device, *args, **kargs):
        super(TensorCP, self).__init__(aabb, device, *args, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.grid_size, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.grid_size, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, grid_size, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, grid_size[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=self.align_corners))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=self.align_corners))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.grid_size == self.grid_size):
            t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total

class TensorVM(TensorVoxelBase):
    def __init__(self, aabb, device, *args, **kargs):
        super(TensorVM, self).__init__(aabb, device, *args, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=self.align_corners).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=self.align_corners).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=self.align_corners).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=self.align_corners).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=self.align_corners))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=self.align_corners))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=self.align_corners)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=self.align_corners)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsampling to {res_target}')

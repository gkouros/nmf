import torch

class PatchSampler:
    def __init__(self, total, batch_size, width, height, device, patch_size=1):
        self.total = total
        self.batch_size = batch_size // patch_size ** 2
        self.width = width
        self.height = height
        self.curr = total
        self.device = device
        self.patch_size = patch_size

        # get total number of patches and patch ids
        W, H = self.width, self.height
        N = int(self.total / W / H)
        P = self.patch_size
        self.total_patches = N * (H - P + 1) * (W - P + 1)
        self.patch_ids = None

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size // self.patch_size ** 2  # B
        self.curr += batch_size
        if self.curr + self.batch_size > self.total_patches:
            self.patch_ids = torch.randperm(self.total_patches, dtype=torch.long, device=self.device)
            self.curr = 0
        patch_ids = self.patch_ids[self.curr : self.curr + batch_size]  # [B]
        N, H, W, P = self.total, self.height, self.width, self.patch_size
        HP = H - P + 1
        WP = W - P + 1
        img_ids = patch_ids // (HP * WP)
        row_ids = patch_ids % (HP * WP) // WP
        col_ids = patch_ids % (HP * WP) % WP

        # get the ids of the left upper corners of the patches
        patch_LUC_ids = img_ids * H * W + row_ids * W + col_ids  # [B]

        # get the id offsets for all patch members
        patch_offsets = torch.arange(P, dtype=torch.long, device=self.device)[..., None].repeat([1, P]) * W + \
            torch.arange(P, dtype=torch.long, device=self.device)  # [P,P]

        # get all the ids organized in patches
        ids = patch_LUC_ids.repeat([P, P, 1]).T + patch_offsets # [B,P,P]

        # return flattened ids ordered in patces
        return ids.ravel()  # [BxPxP]
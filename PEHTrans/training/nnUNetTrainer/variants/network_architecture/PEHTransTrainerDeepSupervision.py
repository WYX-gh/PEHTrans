from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn,MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.network_architecture.PEHTrans import TokenSeg
from torch.cuda.amp import GradScaler
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class PEHTransTrainerDeepSupervision(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True  # 启用深度监督
        if self.device.type == 'cuda':
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = None 

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({}, {'batch_dice': self.configuration_manager.batch_dice, 
                                  'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                           use_ignore_label=self.label_manager.ignore_label is not None,
                           dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, 
                          weight_ce=1, weight_dice=1,
                          ignore_label=self.label_manager.ignore_label, 
                          dice_class=MemoryEfficientSoftDiceLoss)

        # 深监督损失
        deep_supervision_scales =self._get_deep_supervision_scales()
        if deep_supervision_scales is not None:
            weights = [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            weights = weights / np.sum(weights)
            loss = DeepSupervisionWrapper(loss, weights)
        else:
            loss = loss

        return loss

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            self.network = TokenSeg(
                inch=self.num_input_channels, 
                outch=self.label_manager.num_segmentation_heads,
                downlayer=3,
                base_channel=32,
                hidden_size=256,
                imgsize=self.configuration_manager.patch_size,  
                TransformerLayerNum=2
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True

            self.print_to_log_file("="*50)
            self.print_to_log_file("Thanks nnUNet" )
            self.print_to_log_file("now using PEHTrans model")
            self.print_to_log_file("="*50)
        
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device) for t in target]
        else:
            target = target.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=(self.device.type == 'cuda')):
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            #self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device) for t in target]
        else:
            target = target.to(self.device)

        with torch.no_grad():
            output = self.network(data)
            l = self.loss(output, target)

        axes = [0] + list(range(2, output[0].ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output[0]) > 0.5).long()
        else:
            output_seg = output[0].argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output[0].shape, device=output[0].device)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target[0], axes=axes)
        tp_hard = tp.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

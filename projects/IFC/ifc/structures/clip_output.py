from typing import List
import torch
from scipy.optimize import linear_sum_assignment

from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom

from ..util.misc import interpolate


from typing import List
import torch
from scipy.optimize import linear_sum_assignment
from detectron2.utils.memory import retry_if_cuda_oom
from ..util.misc import interpolate

class Videos:
    """
    Memory-efficient structure to support clip-level instance tracking using sliding window
    approach for long videos.
    """
    def __init__(self, num_frames, video_length, num_classes, image_size, device):
        self.num_frames = num_frames
        self.video_length = video_length
        self.device = device
        
        # Parameters for instance tracking
        self.num_max_inst = 50  # max number of instances
        self.match_threshold = 0.05
        self.window_size = 3  # number of clips to keep in memory
        
        self.num_inst = 0
        self.num_clip = 0
        self.saved_idx_set = set()
        
        # Initialize storage for sliding window
        self.saved_logits = torch.zeros(
            (self.window_size, self.num_max_inst, self.video_length, *image_size), 
            dtype=torch.float, 
            device=device
        )
        self.saved_masks = torch.zeros(
            (self.window_size, self.num_max_inst, self.video_length, *image_size), 
            dtype=torch.float, 
            device=device
        )
        self.saved_valid = torch.zeros(
            (self.window_size, self.num_max_inst, self.video_length), 
            dtype=torch.bool, 
            device=device
        )
        self.saved_cls = torch.zeros(
            (self.window_size, self.num_max_inst, num_classes+1), 
            dtype=torch.float, 
            device=device
        )
        
        # Accumulated results
        self.accumulated_logits = torch.zeros(
            (self.num_max_inst, self.video_length, *image_size),
            dtype=torch.float,
            device=device
        )
        self.accumulated_valid = torch.zeros(
            (self.num_max_inst, self.video_length),
            dtype=torch.bool,
            device=device
        )
        self.accumulated_cls = torch.zeros(
            (self.num_max_inst, num_classes+1),
            dtype=torch.float,
            device=device
        )
        self.accumulation_count = torch.zeros(
            (self.num_max_inst, self.video_length),
            dtype=torch.float,
            device=device
        )

    def get_siou(self, input_masks, saved_masks, saved_valid):
        """Calculate spatial IoU between input and saved masks"""
        input_masks = input_masks.flatten(-2)   
        saved_masks = saved_masks.flatten(-2)   

        input_masks = input_masks[None, None]   
        saved_masks = saved_masks.unsqueeze(2)  
        saved_valid = saved_valid[:, :, None, :, None]  

        numerator = saved_masks * input_masks
        denominator = saved_masks + input_masks - saved_masks * input_masks

        numerator = (numerator * saved_valid).sum(dim=(-1, -2))
        denominator = (denominator * saved_valid).sum(dim=(-1, -2))

        siou = numerator / (denominator + 1e-6)
        num_valid_clip = (saved_valid.flatten(2).sum(dim=2) > 0).sum(dim=0)
        siou = siou.sum(dim=0) / (num_valid_clip[..., None] + 1e-6)

        return siou

    def _update_accumulation(self):
        """Update accumulated results with current window"""
        window_idx = (self.num_clip - 1) % self.window_size
        
        # Update accumulated results for the oldest clip in the window
        if self.num_clip > self.window_size:
            valid_mask = self.saved_valid[window_idx]
            self.accumulated_logits += self.saved_logits[window_idx] * valid_mask.unsqueeze(-1).unsqueeze(-1)
            self.accumulation_count += valid_mask.float()
            
            # Update class probabilities
            valid_instances = valid_mask.any(dim=1)
            self.accumulated_cls[valid_instances] += self.saved_cls[window_idx][valid_instances]
            
            # Clear the oldest slot for reuse
            self.saved_logits[window_idx].zero_()
            self.saved_masks[window_idx].zero_()
            self.saved_valid[window_idx].zero_()
            self.saved_cls[window_idx].zero_()

    def update(self, input_clip):
        window_idx = self.num_clip % self.window_size

        # Find intersecting frames 
        inter_input_idx, inter_saved_idx = [], []
        for o_i, f_i in enumerate(input_clip.frame_idx):
            if f_i in self.saved_idx_set:
                inter_input_idx.append(o_i)
                inter_saved_idx.append(f_i)

        # Match instances between clips
        if len(inter_input_idx) > 0 and self.num_inst > 0:
            i_masks = input_clip.mask_probs[:, inter_input_idx]
            s_masks = self.saved_masks[:, :self.num_inst, inter_saved_idx]
            s_valid = self.saved_valid[:, :self.num_inst, inter_saved_idx]

            scores = self.get_siou(i_masks, s_masks, s_valid)
            above_thres = scores > self.match_threshold
            scores = scores * above_thres.float()

            row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)

            existed_idx = []
            for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
                if not is_above:
                    continue

                # Ensure we don't write beyond video length
                frame_indices = [idx for idx in input_clip.frame_idx if idx < self.video_length]

                self.saved_logits[window_idx, r, frame_indices] = input_clip.mask_logits[c, :len(frame_indices)]
                self.saved_masks[window_idx, r, frame_indices] = input_clip.mask_probs[c, :len(frame_indices)]
                self.saved_valid[window_idx, r, frame_indices] = True
                self.saved_cls[window_idx, r] = input_clip.cls_probs[c]
                existed_idx.append(c)

            left_idx = [i for i in range(input_clip.num_instance) if i not in existed_idx]
        else:
            left_idx = list(range(input_clip.num_instance))

        # Add new instances
        if left_idx:
            new_inst_end = min(self.num_inst + len(left_idx), self.num_max_inst)
            if new_inst_end > self.num_inst:
                slice_idx = slice(self.num_inst, new_inst_end)
                left_slice = slice(0, new_inst_end - self.num_inst)

                # Ensure we don't write beyond video length
                frame_indices = [idx for idx in input_clip.frame_idx if idx < self.video_length]

                self.saved_logits[window_idx, slice_idx, frame_indices] = input_clip.mask_logits[left_idx[left_slice], :len(frame_indices)]
                self.saved_masks[window_idx, slice_idx, frame_indices] = input_clip.mask_probs[left_idx[left_slice], :len(frame_indices)]
                self.saved_valid[window_idx, slice_idx, frame_indices] = True
                self.saved_cls[window_idx, slice_idx] = input_clip.cls_probs[left_idx[left_slice]]
                self.num_inst = new_inst_end

        # Update accumulation and status
        self._update_accumulation()
        self.saved_idx_set.update(set(frame_indices))  # Only add valid frame indices
        self.num_clip += 1


    def get_result(self, image_size):
        # Process final window
        for i in range(min(self.window_size, self.num_clip)):
            valid_mask = self.saved_valid[i, :self.num_inst]
            self.accumulated_logits[:self.num_inst] += (
                self.saved_logits[i, :self.num_inst] * valid_mask.unsqueeze(-1).unsqueeze(-1)
            )
            self.accumulation_count[:self.num_inst] += valid_mask.float()
            
            # Update class probabilities for valid instances
            valid_instances = valid_mask.any(dim=1)
            self.accumulated_cls[:self.num_inst][valid_instances] += self.saved_cls[i, :self.num_inst][valid_instances]
        
        # Average accumulated results
        valid_count = torch.clamp(self.accumulation_count[:self.num_inst], min=1.0)
        final_logits = self.accumulated_logits[:self.num_inst] / valid_count.unsqueeze(-1).unsqueeze(-1)
        
        # Resize masks to original image size - only use frames within video length
        final_logits = retry_if_cuda_oom(interpolate)(
            final_logits[:, :self.video_length], size=image_size, mode="bilinear", align_corners=False
        )
        
        # Average class probabilities and determine valid instances
        valid_instances = (valid_count[:, :self.video_length].sum(dim=1) > 0)
        out_cls = self.accumulated_cls[:self.num_inst] / torch.clamp(valid_count.sum(dim=1), min=1.0).unsqueeze(-1)
        out_masks = retry_if_cuda_oom(lambda x: x > 0.0)(final_logits)
        
        return out_cls[valid_instances], out_masks[valid_instances]


class Clips:
    def __init__(self, frame_idx: List[int], results: List[Instances]):
        self.frame_idx = frame_idx
        self.frame_set = set(frame_idx)

        self.classes = results.pred_classes
        self.scores = results.scores
        self.cls_probs = results.cls_probs
        self.mask_logits = results.pred_masks
        self.mask_probs = results.pred_masks.sigmoid()

        self.num_instance = len(self.scores)

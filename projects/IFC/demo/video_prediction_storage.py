import h5py
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Any, NamedTuple, Union
import torch
from scipy import sparse

class CompressedVideoData(NamedTuple):
    """Container for compressed video prediction data"""
    masks_compressed: List[sparse.csr_matrix]  # List of sparse matrices, one per instance
    class_ids: List[int]  # Class IDs for each instance
    class_labels: List[str]  # Class labels for each instance
    confidence_scores: List[float]  # Confidence scores for each instance

def get_class_label(
    class_id: int,
    metadata: Dict[str, Any]
) -> str:
    """
    Get class label for a given class ID using metadata's thing_classes.
    Returns "unknown label" if class_id is invalid.
    """
    detectable_classes = metadata.get("thing_classes", [])
    if isinstance(class_id, (int, np.integer)) and 0 <= class_id < len(detectable_classes):
        return detectable_classes[class_id]
    return "unknown label"

def compress_video_predictions(
    instance_masks: List[torch.Tensor],
    class_ids: List[int],
    confidence_scores: List[float],
    metadata: Dict[str, Any]
) -> CompressedVideoData:
    """
    Compress video predictions using sparse matrix compression.
    
    Args:
        instance_masks: List of mask tensors, each of shape (num_frames, height, width)
        class_ids: List of class IDs for each instance
        confidence_scores: List of confidence scores for each instance
        metadata: Dictionary with metadata including thing_classes
        
    Returns:
        CompressedVideoData object containing compressed predictions
    """
    if not (len(instance_masks) == len(class_ids) == len(confidence_scores)):
        raise ValueError("Number of masks, class IDs, and confidence scores must match")

    masks_compressed = []
    for mask in instance_masks:
        mask_np = mask.cpu().numpy()
        num_frames, height, width = mask_np.shape
        mask_2d = mask_np.reshape(num_frames, -1)
        mask_sparse = sparse.csr_matrix(mask_2d)
        masks_compressed.append(mask_sparse)

    class_labels = [get_class_label(class_id, metadata) for class_id in class_ids]

    return CompressedVideoData(
        masks_compressed=masks_compressed,
        class_ids=class_ids,
        class_labels=class_labels,
        confidence_scores=confidence_scores
    )
    
def save_compressed_data(
    output_dir: Union[str, Path],
    video_id: str,
    compressed_data: CompressedVideoData
) -> None:
    """
    Save compressed video predictions to HDF5 file.
    
    Args:
        output_dir: Directory to store the HDF5 files
        video_id: Unique identifier for the video
        compressed_data: CompressedVideoData object containing predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{video_id}.h5"
    
    with h5py.File(h5_path, 'w') as f:
        masks_group = f.create_group('instance_masks')
        for idx, mask_sparse in enumerate(compressed_data.masks_compressed):
            instance_group = masks_group.create_group(f'instance_{idx}')
            instance_group.create_dataset('data', data=mask_sparse.data, compression='gzip')
            instance_group.create_dataset('indices', data=mask_sparse.indices, compression='gzip')
            instance_group.create_dataset('indptr', data=mask_sparse.indptr, compression='gzip')
            instance_group.attrs['shape'] = mask_sparse.shape
        
        f.attrs['num_instances'] = len(compressed_data.masks_compressed)
        
        class_info = f.create_group('class_info')
        class_info.create_dataset('class_ids', data=np.array(compressed_data.class_ids))
        labels_encoded = json.dumps(compressed_data.class_labels).encode('utf-8')
        class_info.create_dataset('class_labels', data=labels_encoded)

        f.create_dataset(
            'confidence_scores',
            data=np.array(compressed_data.confidence_scores),
            compression='gzip'
        )

def load_frame_prediction(
    predictions_dir: Union[str, Path],
    video_id: str,
    frame_idx: int
) -> Tuple[List[np.ndarray], List[int], List[str], List[float]]:
    """
    Load predictions for a specific frame.
    
    Args:
        predictions_dir: Directory containing the HDF5 files
        video_id: Video identifier
        frame_idx: Index of the frame to load
        
    Returns:
        Tuple of (frame_masks, class_ids, class_labels, confidence_scores)
        where frame_masks is a list of masks for each instance at the specified frame
    """
    h5_path = Path(predictions_dir) / f"{video_id}.h5"
    
    with h5py.File(h5_path, 'r') as f:
        # Get dimensions from the first mask's shape
        first_mask = f['instance_masks/instance_0']
        shape = first_mask.attrs['shape']
        num_frames = shape[0]
        num_pixels = shape[1]
        height = width = int(np.sqrt(num_pixels))
        
        frame_masks = []
        num_instances = f.attrs['num_instances']
        for idx in range(num_instances):
            instance_group = f['instance_masks'][f'instance_{idx}']
            data = instance_group['data'][()]
            indices = instance_group['indices'][()]
            indptr = instance_group['indptr'][()]
            shape = instance_group.attrs['shape']
            
            mask_sparse = sparse.csr_matrix(
                (data, indices, indptr),
                shape=shape
            )
            
            frame_mask = mask_sparse[frame_idx].toarray()
            frame_masks.append(frame_mask.reshape(height, width))
        
        class_ids = f['class_info/class_ids'][()].tolist()
        class_labels = json.loads(f['class_info/class_labels'][()])
        confidence_scores = f['confidence_scores'][()].tolist()
        
    return frame_masks, class_ids, class_labels, confidence_scores

# Example usage
if __name__ == "__main__":
    predictions_dir = "/mnt/data/output_data/IFC/r101_max_inst_50"
    mask_sum = 0
    for i in range(20):
        frame_masks, ids, labels, scores = load_frame_prediction(predictions_dir, "B08_R2_Boden_detectron2_20_frames", i)
        mask_sum += frame_masks[0].sum()
        
    print("Mask sum instance 0: ", mask_sum)
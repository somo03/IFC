import h5py
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Any, NamedTuple
import torch
from scipy import sparse

class CompressedVideoData(NamedTuple):
    """Container for compressed video prediction data"""
    masks_compressed: List[sparse.csr_matrix]  # List of sparse matrices, one per instance
    labels: List[str]  # Labels for each instance, averaged across video frames
    confidence_scores: List[float]  # Scores for each instance, averaged across video frames
    metadata: Dict[str, Any]  # Optional metadata

class VideoPredictionStorage:
    def __init__(self, output_dir: str):
        """
        Initialize storage for video processing predictions.
        
        Args:
            output_dir: Directory to store the HDF5 files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compress_video_predictions(
        self,
        instance_masks: List[torch.Tensor],
        labels: List[str],
        confidence_scores: List[float],
        metadata: Dict[str, Any] = None
    ) -> CompressedVideoData:
        """
        Compress video predictions using sparse matrix compression.
        
        Args:
            instance_masks: List of mask tensors, each of shape (num_frames, height, width)
            labels: List of labels for each instance
            confidence_scores: List of confidence scores for each instance
            metadata: Optional dictionary with additional metadata
            
        Returns:
            CompressedVideoData object containing compressed predictions
        """
        # Validate inputs
        if not (len(instance_masks) == len(labels) == len(confidence_scores)):
            raise ValueError("Number of masks, labels, and confidence scores must match")

        # Compress each instance mask
        masks_compressed = []
        for mask in instance_masks:
            # Convert to numpy and flatten to 2D (frames, pixels)
            mask_np = mask.cpu().numpy()
            num_frames, height, width = mask_np.shape
            mask_2d = mask_np.reshape(num_frames, -1)
            
            # Convert to sparse matrix
            mask_sparse = sparse.csr_matrix(mask_2d)
            masks_compressed.append(mask_sparse)

        return CompressedVideoData(
            masks_compressed=masks_compressed,
            labels=labels,
            confidence_scores=confidence_scores,
            metadata=metadata or {}
        )
        
    def save_compressed_data(
        self,
        video_id: str,
        compressed_data: CompressedVideoData
    ) -> None:
        """
        Save compressed video predictions to HDF5 file.
        
        Args:
            video_id: Unique identifier for the video
            compressed_data: CompressedVideoData object containing predictions
        """
        h5_path = self.output_dir / f"{video_id}.h5"
        
        with h5py.File(h5_path, 'w') as f:
            # Create a group for instance masks
            masks_group = f.create_group('instance_masks')
            
            # Store each compressed instance mask
            for idx, mask_sparse in enumerate(compressed_data.masks_compressed):
                instance_group = masks_group.create_group(f'instance_{idx}')
                
                # Store sparse matrix components
                instance_group.create_dataset('data', data=mask_sparse.data, compression='gzip')
                instance_group.create_dataset('indices', data=mask_sparse.indices, compression='gzip')
                instance_group.create_dataset('indptr', data=mask_sparse.indptr, compression='gzip')
                instance_group.attrs['shape'] = mask_sparse.shape
            
            # Store number of instances
            f.attrs['num_instances'] = len(compressed_data.masks_compressed)
            
            # Store original dimensions in metadata
            shape_metadata = compressed_data.metadata.copy()
            if 'mask_dims' not in shape_metadata:
                # Get dimensions from first mask if available
                if compressed_data.masks_compressed:
                    num_frames = compressed_data.masks_compressed[0].shape[0]
                    num_pixels = compressed_data.masks_compressed[0].shape[1]
                    height = int(np.sqrt(num_pixels))  # Assuming square frames for this example
                    shape_metadata['mask_dims'] = {
                        'num_frames': num_frames,
                        'height': height,
                        'width': height
                    }
            
            # Store labels
            labels_encoded = json.dumps(compressed_data.labels).encode('utf-8')
            f.create_dataset('labels', data=labels_encoded)
            
            # Store confidence scores
            f.create_dataset(
                'confidence_scores',
                data=np.array(compressed_data.confidence_scores),
                compression='gzip'
            )
            
            # Store metadata
            metadata_encoded = json.dumps(shape_metadata).encode('utf-8')
            f.create_dataset('metadata', data=metadata_encoded)

    def load_frame_prediction(
        self,
        video_id: str,
        frame_idx: int
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """
        Load predictions for a specific frame.
        
        Args:
            video_id: Video identifier
            frame_idx: Index of the frame to load
            
        Returns:
            Tuple of (frame_masks, labels, confidence_scores)
            where frame_masks is a list of masks for each instance at the specified frame
        """
        h5_path = self.output_dir / f"{video_id}.h5"
        
        with h5py.File(h5_path, 'r') as f:
            num_instances = f.attrs['num_instances']
            metadata = json.loads(f['metadata'][()])
            mask_dims = metadata['mask_dims']
            height = mask_dims['height']
            width = mask_dims['width']
            
            # Load masks for the specified frame
            frame_masks = []
            for idx in range(num_instances):
                instance_group = f['instance_masks'][f'instance_{idx}']
                
                # Reconstruct sparse matrix
                data = instance_group['data'][()]
                indices = instance_group['indices'][()]
                indptr = instance_group['indptr'][()]
                shape = instance_group.attrs['shape']
                
                mask_sparse = sparse.csr_matrix(
                    (data, indices, indptr),
                    shape=shape
                )
                
                # Extract the requested frame and reshape
                frame_mask = mask_sparse[frame_idx].toarray()
                frame_masks.append(frame_mask.reshape(height, width))
            
            # Load labels and confidence scores
            labels = json.loads(f['labels'][()])
            confidence_scores = f['confidence_scores'][()].tolist()
            
        return frame_masks, labels, confidence_scores

# Example usage
if __name__ == "__main__":
    # Initialize storage
    storage = VideoPredictionStorage("output/predictions")
    
    # Create example predictions
    num_frames = 10
    height = width = 640  # Square frames for simplicity
    
    # Create 3 instance masks (3 different objects detected across video)
    # Using sparse masks (mostly zeros with some ones) to demonstrate compression
    instance_masks = []
    for _ in range(3):
        mask = torch.zeros((num_frames, height, width), dtype=torch.bool)
        # Add some random true values (sparse)
        for f in range(num_frames):
            x = torch.randint(0, width, (100,))
            y = torch.randint(0, height, (100,))
            mask[f, y, x] = True
        instance_masks.append(mask)
    
    # Labels and confidence scores for each instance
    labels = ["person", "car", "dog"]
    confidence_scores = [0.95, 0.87, 0.92]
    metadata = {"fps": 30, "resolution": f"{width}x{height}"}
    
    # First compress the predictions
    compressed_data = storage.compress_video_predictions(
        instance_masks,
        labels,
        confidence_scores,
        metadata
    )
    
    # Then save the compressed data
    storage.save_compressed_data("video_001", compressed_data)
    
    # Load predictions for frame 0
    frame_masks, video_labels, video_scores = storage.load_frame_prediction("video_001", 0)
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import rasterio
import csv

# Paths to the images and masks directories
images_dir = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/train_imgs"
masks_dir = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/train_masks"

# List to hold the image and mask pairs
data = []

## DATA


# Get sorted lists of image and mask file names
image_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_IMG.tif"))
mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_MASK.tif"))

# Ensure the image and mask filenames align properly
for img_file in image_files:
    # Generate the corresponding mask file name
    mask_file = img_file.replace("_IMG.tif", "_MASK.tif")
    
    # Check if the corresponding mask exists
    if mask_file in mask_files:
        # Create full paths for the image and mask
        image_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        # Append to the data list
        data.append({"image": image_path, "annotation": mask_path})
    else:
        print(f"Warning: No corresponding mask for image {img_file}")


## TEST_DATA

images_dir = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/val_imgs"
masks_dir = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/sam2_fasteo/GhanaMining3bands_final/val_masks"

# List to hold the image and mask pairs
test_data = []

# Get sorted lists of image and mask file names
image_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_IMG.tif"))
mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_MASK.tif"))

# Ensure the image and mask filenames align properly
for img_file in image_files:
    # Generate the corresponding mask file name
    mask_file = img_file.replace("_IMG.tif", "_MASK.tif")
    
    # Check if the corresponding mask exists
    if mask_file in mask_files:
        # Create full paths for the image and mask
        image_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        # Append to the data list
        test_data.append({"image": image_path, "annotation": mask_path})
    else:
        print(f"Warning: No corresponding mask for image {img_file}")



print("Len(data)")
print(len(data))
print("Len(test_data)")
print(len(test_data))

NUM_EPOCHS = 10

def read_batch_test(data):
    ent  = data[np.random.randint(len(data))]
    with rasterio.open(ent["image"]) as src:
        Img = np.dstack([src.read(1), src.read(2), src.read(3)])
    with rasterio.open(ent["annotation"]) as src:
        ann_map = src.read(1)
    inds = np.unique(ann_map)
    points = []
    masks = []
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def read_batch(data, idx):
    ent  = data[idx]
    with rasterio.open(ent["image"]) as src:
        Img = np.dstack([src.read(1), src.read(2), src.read(3)])
    with rasterio.open(ent["annotation"]) as src:
        ann_map = src.read(1)
    inds = np.unique(ann_map)
    points = []
    masks = []
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

# Load model
sam2_checkpoint = "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0016/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()

def evaluate_model(test_data):
    """
    Function to evaluate the model on test_data and return the average IoU.
    """
    total_iou = 0
    count = 0
    
    with torch.no_grad():  # Disable gradient calculations
        for test_sample in test_data:
            image, mask, input_point, input_label = read_batch_test([test_sample])  # Load test batch
            if mask.shape[0] == 0: continue  # Skip empty batches
            
            predictor.set_image(image)  # Apply SAM image encoder
            
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None)
            
            batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features)
            
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map into probability map
            
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            
            # Calculate IoU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
            iou = inter / (union + 1e-7)  # Add a small epsilon to avoid division by zero
            
            total_iou += iou.mean().item()  # Accumulate mean IoU for each test sample
            count += 1
    
    return total_iou / count if count > 0 else 0  # Return the average IoU

# Training loop
for itr in range(NUM_EPOCHS * len(data)): #2967: all training images (len(data)), N * len(data) = N epochs
    with torch.cuda.amp.autocast():
        image, mask, input_point, input_label = read_batch(data, (itr % len(data))) #previously -> np.random.randint(len(data))
        if mask.shape[0] == 0: continue
        predictor.set_image(image)
        
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
        
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0), image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
        
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05
        
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #if itr % 1000 == 0 and itr != 0:
        #    torch.save(predictor.model.state_dict(), "model_cocoa.torch")
        #    print("Model saved at iteration", itr)
        
        if itr == 0: mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print(f"Step: {itr}, Training Accuracy (IoU): {mean_iou}")

# Evaluate on test data every 1000 iterations
torch.save(predictor.model.state_dict(), "model_cocoa_10epochs.torch")
print("Model saved at end of training")
test_iou = evaluate_model(test_data)
print(f"Test Accuracy (IoU) at iteration {itr} (10 epochs): {test_iou}")
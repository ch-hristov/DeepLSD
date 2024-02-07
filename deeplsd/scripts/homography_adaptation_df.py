"""
Run the homography adaptation for all images in a given folder
to regress and aggregate line distance function maps.
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm
from pytlsd import lsd
from afm_op import afm
from joblib import Parallel, delayed

from ..datasets.utils.homographies import sample_homography, warp_lines
from ..datasets.utils.data_augmentation import random_contrast


homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}


def ha_df(img, num=100, border_margin=3, min_counts=5):
    """ Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]
    size = (w, h)
    df_maps, angles, closests, counts = [], [], [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (border_margin * 2, border_margin * 2))
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'),
                       axis=-1)
    raster_lines = np.zeros_like(img)

    # Loop through all the homographies
    for i in range(num):
        # Generate a random homography
        if i == 0:
            H = np.eye(3)
        else:
            H = sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)
        
        # Warp the image
        warped_img = cv2.warpPerspective(img, H, size,
                                         borderMode=cv2.BORDER_REPLICATE)
        
        # Regress the DF on the warped image
        warped_lines = lsd(warped_img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        
        # Warp the lines back
        lines = warp_lines(warped_lines, H_inv)
        
        # Get the DF and angles
        num_lines = len(lines)
        cuda_lines = torch.from_numpy(lines[:, :, [1, 0]].astype(np.float32))
        cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda()
        offset = afm(
            cuda_lines,
            torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w)[0]
        offset = offset[0].permute(1, 2, 0).cpu().numpy()[:, :, [1, 0]]
        closest = pix_loc + offset
        df = np.linalg.norm(offset, axis=-1)
        angle = np.mod(np.arctan2(
            offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi)
        
        df_maps.append(df)
        angles.append(angle)
        closests.append(closest)
        
        # Compute the valid pixels
        count = cv2.warpPerspective(np.ones_like(img), H_inv, size,
                                    flags=cv2.INTER_NEAREST)
        count = cv2.erode(count, kernel)
        counts.append(count)
        raster_lines += (df < 1).astype(np.uint8) * count 
        
    # Compute the median of all DF maps, with counts as weights
    df_maps, angles = np.stack(df_maps), np.stack(angles)
    counts, closests = np.stack(counts), np.stack(closests)
    
    # Median of the DF
    df_maps[counts == 0] = np.nan
    avg_df = np.nanmedian(df_maps, axis=0)

    # Median of the closest
    closests[counts == 0] = np.nan
    avg_closest = np.nanmedian(closests, axis=0)

    # Median of the angle
    circ_bound = (np.minimum(np.pi - angles, angles)
                  * counts).sum(0) / counts.sum(0) < 0.3
    angles[:, circ_bound] -= np.where(
        angles[:, circ_bound] > np.pi /2,
        np.ones_like(angles[:, circ_bound]) * np.pi,
        np.zeros_like(angles[:, circ_bound]))
    angles[counts == 0] = np.nan
    avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi)

    # Generate the background mask and a saliency score
    raster_lines = np.where(raster_lines > min_counts, np.ones_like(img),
                            np.zeros_like(img))
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(float)

    return avg_df, avg_angle, avg_closest[:, :, [1, 0]], bg_mask


def ha_df_with_lines(img, lines, num = 100, border_margin=3, min_counts=5):
    """ Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        lines: a list of line segments in the image, each line segment is a tuple of four coordinates (x1, y1, x2, y2).
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2] # get the height and width of the image
    expanded_lines = []

    for entry in lines:
        new_entry_a = [int(entry[0] * w), int(entry[1] * h) ]
        new_entry_b = [new_entry_a[0]+ int(entry[2] * w), new_entry_a[1] + int(entry[3] * h)]
        expanded_lines.append([new_entry_a[0],new_entry_a[1], new_entry_b[0],new_entry_b[1]])    
    np_lines = np.array(expanded_lines)
        
    size = (w, h) # define the size of the image as a tuple
    df_maps, angles, closests, counts = [], [], [], [] # initialize empty lists to store the distance function maps, angles, closest points, and counts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (border_margin * 2, border_margin * 2)) # create a kernel for erosion
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'),
                       axis=-1) # create a pixel location map
    raster_lines = np.zeros_like(img) # create a zero matrix to store the rasterized lines

    # Loop through all the homographies
    for i in tqdm(range(num), "Computing homography", len(range(num))):
        # Generate a random homography
        if i == 0:
            H = np.eye(3) # use the identity matrix as the first homography
        else:
            H = sample_homography(img.shape, **homography_params) # sample a random homography from the predefined parameters
        H_inv = np.linalg.inv(H) # compute the inverse of the homography
        
        # Warp the lines
        warped_lines = warp_lines(np_lines, H) # apply the homography to the input lines
        
        # Get the DF and angles
        num_lines = len(warped_lines) # get the number of warped lines
        cuda_lines = torch.from_numpy(warped_lines[:, :, [1, 0]].astype(np.float32)) # convert the warped lines to a torch tensor
        cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda() # reshape and move the tensor to cuda
        offset = afm(
            cuda_lines,
            torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w)[0] # use the AFM function to compute the offset map
        offset = offset[0].permute(1, 2, 0).cpu().numpy()[:, :, [1, 0]] # permute, move to cpu, and convert the offset map to a numpy array
        closest = pix_loc + offset # compute the closest point map by adding the pixel location map and the offset map
        df = np.linalg.norm(offset, axis=-1) # compute the distance function map by taking the norm of the offset map
        angle = np.mod(np.arctan2(
            offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi) # compute the angle map by taking the arctan2 of the offset map and adding pi/2
        
        df_maps.append(df) # append the distance function map to the list
        angles.append(angle) # append the angle map to the list
        closests.append(closest) # append the closest point map to the list
        
        # Compute the valid pixels
        count = cv2.warpPerspective(np.ones_like(img), H_inv, size,
                                    flags=cv2.INTER_NEAREST) # warp a matrix of ones with the inverse homography to get the valid pixels
        count = cv2.erode(count, kernel) # erode the valid pixels with the kernel
        counts.append(count) # append the count map to the list
        raster_lines += (df < 1).astype(np.uint8) * count # update the rasterized lines by adding the pixels that are close to the lines and valid
        
    # Compute the median of all DF maps, with counts as weights
    df_maps, angles = np.stack(df_maps), np.stack(angles) # stack the distance function maps and angle maps along a new axis
    counts, closests = np.stack(counts), np.stack(closests) # stack the count maps and closest point maps along a new axis
    
    # Median of the DF
    df_maps[counts == 0] = np.nan # set the pixels that are not valid to nan
    avg_df = np.nanmedian(df_maps, axis=0) # compute the median of the distance function maps along the first axis, ignoring nan values

    # Median of the closest
    closests[counts == 0] = np.nan # set the pixels that are not valid to nan
    avg_closest = np.nanmedian(closests, axis=0) # compute the median of the closest point maps along the first axis, ignoring nan values

    # Median of the angle
    circ_bound = (np.minimum(np.pi - angles, angles)
                  * counts).sum(0) / counts.sum(0) < 0.3 # compute a boolean mask for the pixels that are close to the circular boundary
    
    angles[:, circ_bound] -= np.where(
        angles[:, circ_bound] > np.pi /2,
        np.ones_like(angles[:, circ_bound]) * np.pi,
        np.zeros_like(angles[:, circ_bound])) # adjust the angles for the pixels that are close to the circular boundary by subtracting pi or zero
    
    angles[counts == 0] = np.nan # set the pixels that are not valid to nan
    avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi) # compute the median of the angle maps along the first axis, ignoring nan values, and take the modulo of pi

    # Generate the background mask and a saliency score
    raster_lines = np.where(raster_lines > min_counts, np.ones_like(img),
                            np.zeros_like(img)) # create a binary mask for the pixels that are activated by more than min_count
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8)) # dilate the mask with a square kernel
    bg_mask = (1 - raster_lines).astype(float) # create a background mask by subtracting the rasterized lines from one

    return avg_df, avg_angle, avg_closest[:, :, [1, 0]], bg_mask # return the aggregated distance function maps, angle maps, closest point maps, and background mask

def process_image(img_path, randomize_contrast, num_H, output_folder, lines):
    img = cv2.imread(img_path, 0)
    out_path = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, out_path) + '.hdf5'
    if not os.path.exists(out_path):
      if randomize_contrast is not None:
          img = randomize_contrast(img)
      
      # Run homography adaptation
      df, angle, closest, bg_mask = ha_df_with_lines(img,lines= lines,num=num_H)

      # Check if the hdf5 file already exists in the output folder
    
    
      try:
        # Save the DF in a hdf5 file
        print("Saving image file to " + out_path)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("df", data=df.flatten())
            f.create_dataset("line_level", data=angle.flatten())
            f.create_dataset("closest", data=closest.flatten())
            f.create_dataset("bg_mask", data=bg_mask.flatten())
      except Exception as ex:
        print(str(ex))
    else:
        print("The hdf5 file already exists. Skipping this image.")

def export_ha(images_list, output_folder, lines_annotation_folder, num_H=100,
              rdm_contrast=False, n_jobs=1):
    # Parse the data
    with open(images_list, 'r') as f:
        image_files = f.readlines()

    d = {}
    
    for dir, _, files in os.walk(lines_annotation_folder):
            for file in files:
                ann_base = os.path.basename(file)[0]
                with open(os.path.join(dir, file), 'r') as annotations:
                    all_annotations= annotations.readlines()
                    entries = [x.replace('\n','').split(' ') for x in all_annotations]

                    if ann_base not in d:
                        d[ann_base] = []
                    for item in entries:
                        d[ann_base].append([float(item[1]), float(item[2]), float(item[3]), float(item[4])])
                

    image_files = [path.strip('\n') for path in image_files]
    
    # Random contrast object
    randomize_contrast = random_contrast() if rdm_contrast else None
    
    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_image)(
        img_path, randomize_contrast, num_H, output_folder, d[os.path.basename(img_path)[0]])
                                            for img_path in tqdm(image_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_list', type=str,
                        help='Path to a txt file containing the image paths.')
  
    parser.add_argument('output_folder', type=str, help='Output folder.')
    parser.add_argument('lines_annotation_folder', type=str, 
                        help='Path to where the .txt YOLO formatted lines are stored.')
    parser.add_argument('--num_H', type=int, default=100,
                        help='Number of homographies used during HA.')
    parser.add_argument('--random_contrast', action='store_true',
                        help='Add random contrast to the images (disabled by default).')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs to run in parallel.')
    args = parser.parse_args()

    export_ha(args.images_list, args.output_folder, args.lines_annotation_folder, args.num_H,
              args.random_contrast, args.n_jobs)

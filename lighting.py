from sklearn.cluster import KMeans
from ddim import *
from utils import single_depatchify, gaussDnew

MIN_PIXELS = 100
PATCH_SIZE = 16  # Define a constant for patch size


def infer_light_direction(normals, intensities):
    """
    Estimate light direction using least squares.
    Args:
        normals (np.ndarray): Normal vectors (N x 3).
        intensities (np.ndarray): Observed intensities (N x 1).
    Returns:
        tuple: Light direction (3D vector) and estimation error.
    """
    x, _, _, _ = np.linalg.lstsq(normals, intensities, rcond=None)
    light_source_direction = x
    predicted_intensities = np.dot(normals, light_source_direction)
    error = np.linalg.norm(predicted_intensities - intensities)
    return light_source_direction, error


def patch_lighting_guess(pred_normals, batch_img_patch, res, masks = None, normalize = True, min_num = 30):
    """
    Estimate light direction for patches using least squares.
    """
    b = pred_normals.shape[0]
    errs = np.zeros((b))
    ls = np.zeros((b, 3), dtype = np.float32)
    used = np.zeros((b), dtype = bool)

    for i in range(b):
      curr_pred = pred_normals[i]
      curr_img = batch_img_patch[i]
  
      if normalize:
        curr_pred = normalize_normals_torch(curr_pred)
    
      curr_mask = masks[i][0] if masks is not None else torch.ones_like(curr_img[0], dtype=torch.bool)

      if curr_mask.sum() >= min_num:
        used[i] = True
        pred_vec = ((curr_pred.permute(1, 2, 0))[curr_mask]).view(-1, 3).float().numpy()
        img_vec = ((curr_img.permute(1, 2, 0))[curr_mask]).view(-1, 1).float().numpy()
        l, err = infer_light_direction(pred_vec, img_vec)

        ls[i] = l[:, 0]
        ls[i] = ls[i] / np.linalg.norm(ls[i] + 1e-4)
        
        # robustness check:
        if abs(ls[i][0]) + abs(ls[i][1]) + abs(ls[i][2]) <= 1e-3:
          used[i] = False
        errs[i] = err

    return ls, errs, used


def patch_lighting_guess_torch(pred_normals, batch_img_patch, res, masks = None, min_num = 30):
    b = pred_normals.shape[0]
    errs = torch.zeros((b))
    ls = torch.zeros((b, 3))
    used = np.zeros((b), dtype = bool)

    for i in range(b):
      if masks == None:
        # if no mask, use all information
        curr_mask = torch.ones_like(batch_img_patch[i][0], dtype=torch.bool)
      else:
        curr_mask = masks[i][0]

      if curr_mask.sum() >= min_num:
        used[i] = True
        A = (((pred_normals[i]).permute(1, 2, 0))[curr_mask]).view(-1, 3)
        B = (batch_img_patch[i].permute(1, 2, 0))[curr_mask].view(-1, 1)
        X = torch.linalg.pinv(A) @ B
        ls[i] = X[:, 0]
        errs[i] = torch.dist(B, A @ X)
      else:
        print('not enough information: ', i)

    return ls, errs, used


def labels_fill_unused(labels, used):
    """
    Fill unused patch labels with a default value.
    """  
    all_labels = np.ones(len(used), dtype = int) * 2 # unused-patches get label 2.
    count = 0
    for i in range(len(used)):
        if used[i]:
            all_labels[i] = labels[count]
            count += 1
    return all_labels

  
def compare_dist_convex_concave(input, anchor, threshold = 0.1):
    """
    Compare convex / concave choice to minimize dist (l1, l2, l3) and (l1*, l2*, l3*) - majority cluster center
    """
    x1, y1, z1 = input; x2, y2, z2 = anchor

    assert abs(z1) > 1e-5, "Encountered (0, 0, 0) case"

    vectors = [(x1, y1, z1), (-x1, -y1, z1)]
    coeff = [(1, 1), (-1, -1)]

    count = 0

    # Convert to numpy arrays for easier computation
    target_vector = np.array([x2, y2, z2])
    vectors = np.array(vectors)

    # Calculate cosine similarity between each vector and the target vector
    similarities = np.dot(vectors, target_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector))

    # Find the index of the vector with the highest cosine similarity
    closest_index = np.argmax(similarities)

    if similarities[closest_index] - similarities[1-closest_index] < threshold:
      closest_index = 0 # only flip if lighting variance reduce a lot

    if coeff[closest_index] != (1, 1):
      count += 1

    return vectors[closest_index], coeff[closest_index], count


def flip_by_cluster_label(normal_patches, light_preds, labels, centers, flip_mask = None):
    """
    Flip normal patches based on clustering labels.
    """
    c0_count = (labels == 0).sum(); c1_count = (labels == 1).sum()
    if c0_count >= c1_count:
      c = centers[0]; to_flip = 1
    else:
      c = centers[1]; to_flip = 0

    new_normals = normal_patches.copy()
    for i in range(len(normal_patches)):
      if labels[i] == to_flip:
        # vec, coeff = compare_dist_fourway(light_preds[i], c)
        vec, coeff, count = compare_dist_convex_concave(light_preds[i], c)
        if flip_mask != None:
            curr_mask = flip_mask[i][0]
            (new_normals[i])[0, curr_mask] = (new_normals[i])[0, curr_mask] * coeff[0]
            (new_normals[i])[1, curr_mask] = (new_normals[i])[1, curr_mask] * coeff[1]
        else:
            new_normals[i][0] = new_normals[i][0] * coeff[0]
            new_normals[i][1] = new_normals[i][1] * coeff[1]

    return new_normals, count


def lighting_est_mask(img_orig, seg_mask = None):
    # apply lighting estimation only to high intensity grad regions (> median)
    im_d = gaussDnew(img_orig, 6, 1, 0, 1)
    l_mask = (im_d[1]**2 + im_d[2]**2) > np.median(im_d[1]**2 + im_d[2]**2)
    seg_mask = img_orig >= 1e-5
    
    if seg_mask is not None:
      # select high grad regions within the mask
      Ix = im_d[1]; Iy = im_d[2]
      l_mask = (Ix**2 + Iy**2) >= np.median(Ix[seg_mask]**2 + Iy[seg_mask]**2)
      l_mask = np.bitwise_and(l_mask, seg_mask)

    l_mask_patch = batch_patchify(torch.tensor(l_mask, dtype = torch.bool).unsqueeze(0).unsqueeze(0), 16, 16)
    return l_mask_patch, l_mask


def pred_flip_with_lighting(res, batch_img_patch, batch_nml_pred, img_orig, method = 'lstsq', min_pixels = 80, seg_mask = None):
    """
    Plug-and-play module, can be added after denoising, or in between denoising using predicted x0
    """
    assert res == img_orig.shape[0]
    l_mask_patch, _ = lighting_est_mask(img_orig, seg_mask)

    if method == 'inverse': # matrix inverse
        inferred_ls, errs, used = patch_lighting_guess_torch(torch.tensor(batch_nml_pred), torch.tensor(batch_img_patch), res, l_mask_patch, min_pixels)
    elif method == 'lstsq': # numpy linalg least square
        inferred_ls, errs, used = patch_lighting_guess(torch.tensor(batch_nml_pred), torch.tensor(batch_img_patch), res, l_mask_patch, False, min_pixels)
    else:
        raise(NotImplementedError)

    kmeans = KMeans(n_clusters=2).fit(inferred_ls[used])
    label_filled = labels_fill_unused(kmeans.labels_, used)

    flip_mask = batch_patchify(torch.tensor(seg_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0), PATCH_SIZE, PATCH_SIZE) if seg_mask is not None else None
    flipped, count = flip_by_cluster_label(batch_nml_pred, inferred_ls, label_filled, kmeans.cluster_centers_, flip_mask)
    flipped_global = single_depatchify(flipped, PATCH_SIZE, int(res / PATCH_SIZE), int(res / PATCH_SIZE))

    return flipped, flipped_global, inferred_ls, count

       
def lighting_guided_DDIM_restart(res, batch_img_patch, prev_pred, subsampled_img, method = 'lstsq', min_pixels = MIN_PIXELS, seg_mask = None, flip_restart_ts = 232):
    '''
    Combined method: (1) Regular DDIM, (2) Lighting Consistency Flip, (3) Re-denoising
    '''
    flipped, flipped_global, inferred_ls, count = pred_flip_with_lighting(res, batch_img_patch, prev_pred, subsampled_img, 
                                                                          method = method, min_pixels = min_pixels, seg_mask = seg_mask)
    
    back_time = flip_restart_ts
    total_num = (int(res / PATCH_SIZE))**2
    t = torch.full((total_num,), back_time, dtype=torch.long)
    nmls_patch_noisy = q_sample(torch.tensor(flipped), t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    
    return nmls_patch_noisy, flipped_global, count
    
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_sift_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < 0.8 * n.distance]
    return good

def fundamental_matrix(kp1, kp2, matches):
    if len(matches) < 8:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask.flatten() if mask is not None else None

def compute_homography(kp1, kp2, inliers):
    if len(inliers) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H, mask.flatten() if mask is not None else None

def main(input_dir, output_csv):
    os.chdir(input_dir)
    img_name_list = [f for f in os.listdir() if f.lower().endswith('.jpg')]
    keypoints_data = {}
    
    results = []
    
    for img_name in img_name_list:
        kp, desc = extract_sift_keypoints(img_name)
        keypoints_data[img_name] = (kp, desc)
    
    for i in range(len(img_name_list)):
        for j in range(i + 1, len(img_name_list)):
            img1_name, img2_name = img_name_list[i], img_name_list[j]
            kp1, desc1 = keypoints_data[img1_name]
            kp2, desc2 = keypoints_data[img2_name]
            
            if desc1 is None or desc2 is None:
                continue
            
            matches = match_keypoints(desc1, desc2)
            fraction1 = len(matches) / len(kp1) if kp1 else 0
            fraction2 = len(matches) / len(kp2) if kp2 else 0
            
            F, mask = fundamental_matrix(kp1, kp2, matches)
            inliers = [matches[i] for i in range(len(matches)) if mask is not None and mask[i]]
            inlier_ratio = len(inliers) / len(matches) if matches else 0
            
            H, homoMask = compute_homography(kp1, kp2, inliers)
            newInliers = [inliers[i] for i in range(len(inliers)) if homoMask is not None and homoMask[i]]
            newInlierRatio = len(newInliers) / len(inliers) if inliers else 0
            
            decision = "Accepted" if len(newInliers) >= 25 and newInlierRatio >= 0.3 else "Rejected"
            
            results.append({
                "Image Pair": f"{img1_name} - {img2_name}",
                "Keypoints (Img1)": len(kp1),
                "Keypoints (Img2)": len(kp2),
                "Matches": len(matches),
                "Match Ratio (Img1)": fraction1,
                "Match Ratio (Img2)": fraction2,
                "Fundamental Inliers": len(inliers),
                "Inlier Ratio": inlier_ratio,
                "New Inliers": len(newInliers),
                "New Inlier Ratio": newInlierRatio,
                "Final Decision": decision
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

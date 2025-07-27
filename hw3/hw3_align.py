import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_sift_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()  # Create SIFT detector
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2):
    '''
    matches based on desc from the two imgs. Code is based on 
    https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    Uses brute force method and applies ratio test by D.Lowe
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    return good

def fundamental_matrix(kp1, kp2, matches):
    if len(matches) < 8:
        return None, None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    return F, mask.flatten()

def compute_homography(kp1, kp2, inliers):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    return H, mask


def create_mosaic(img1, img2, H):
    #get the dimensions of img2
    h2, w2 = img2.shape[:2]
    #warp img1 onto img2's coordinate space
    warped_img1 = cv2.warpPerspective(img1, H, (w2, h2))
    #create a mask of non-black pixels in the warped image
    mask = warped_img1 > 0 
    #start with img2 as the base
    mosaic = np.copy(img2)    
    #blend the two images by averaging overlapping areas
    mosaic[mask] = 0.5 * mosaic[mask] + 0.5 * warped_img1[mask]

    return mosaic


def main(arg1, arg2):
    output_file = open(arg2, 'w')
    #opening dir and getting imgs
    start_cwd = os.getcwd()
    os.chdir(arg1)
    img_name_list = os.listdir('./')
    img_name_list = [name for name in img_name_list if 'jpg' in name.lower()]
    
    keypoints_data = {}

    #q1: outputing num of keypoints in each img using sift
    output_file.write(f"Keypoint counts using SIFT\n")
    for img_name in img_name_list:
        kp, desc = extract_sift_keypoints(img_name)
        keypoints_data[img_name] = (kp, desc)
        output_file.write(f"{img_name}: {len(kp)}\n")

    #q2: match keypoints between images 
    for i in range(len(img_name_list)):
        for j in range(i+1, len(img_name_list)):
            kp1, desc1 = keypoints_data[img_name_list[i]]
            kp2, desc2 = keypoints_data[img_name_list[j]]

            if desc1 is not None and desc2 is not None:
                matches = match_keypoints(desc1, desc2)

                #a. fraction of matches:
                fraction1 = len(matches) / len(kp1)
                fraction2 = len(matches) / len(kp2)
                output_file.write(f"\nMatches between {img_name_list[i]} and {img_name_list[j]}: {len(matches)}\n")
                output_file.write(f"Fraction of keypoints matched: {fraction1:.4f}, {fraction2:.4f}\n")

                #b. img with lines of matching points. Used drawMatchesKnn instead of drawMatches
                img1 = cv2.imread(img_name_list[i], cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img_name_list[j], cv2.IMREAD_GRAYSCALE)
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img3)
                plt.title("Keypoint Comparison")
                plt.show()
                output_file.write("Image of Keypoint Comparison: \n")

                #q3 threshold for too small of similarity, going to use fractions from q2 part a
                if len(matches) < 100 or fraction1 < 0.05 or fraction2 < 0.05:
                    output_file.write("Failed threshold of matching keypoints\n")
                    output_file.write("Too little matching keypoints\n")
                    continue
                else:
                    output_file.write("Passed threshold of matching keypoints\n")
                    output_file.write("Amount of matching keypoints meets threshold\n")

                #q4 fundamental matrix found using cv2.findFundamentalMat and method ransac
                F, mask = fundamental_matrix(kp1, kp2, matches)
                inliers = [matches[i] for i in range(len(matches)) if mask[i]]

                #q4 part a
                inlier_ratio = len(inliers)/len(matches)
                output_file.write(f"\nInlier count between {img_name_list[i]} and {img_name_list[j]}: {len(inliers)}\n")
                output_file.write(f"Ratio of inliers to initial match: {inlier_ratio}\n")

                #q4 part b
                img4 = cv2.drawMatches(img1,kp1,img2,kp2,inliers,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img4)
                plt.title("Fundamental Filter")
                plt.show()
                output_file.write("Image consistent with Fundamental Matrix: \n")

                # #q4 part c
                # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                # pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                # # Compute the epipolar lines in the second image for points in the first image
                # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
                # lines2 = lines2.reshape(-1, 3)

                # img1_with_lines, img2_with_lines = img1.copy(), img2.copy()
                

                #q5 too little inliers or too small of a ratio, skip
                if len(inliers) < 25 or inlier_ratio < 0.3:
                    output_file.write("Failed threshold for inliers:\n")
                    output_file.write("Too little inliers or too small of a ratio, images do not match\n")
                    continue
                else:
                    output_file.write("Passed threshold for inliers:\n")
                    output_file.write("Contains enough inliers to be considered as matching images\n")

                #q6 finding the homography matrix
                H, homoMask = compute_homography(kp1, kp2, inliers)
                #output_file.write("Estimated Homography Matrix:\n", H)
                newInliers = [inliers[i] for i in range(len(inliers)) if mask[i]]
                newInlierRatio = len(newInliers)/len(inliers)
                output_file.write(f"\nNew inlier count using homography between {img_name_list[i]} and {img_name_list[j]}: {len(newInliers)}\n")
                output_file.write(f"Ratio of new inliers to fundamental matrix inliers: {newInlierRatio}\n")
                img5 = cv2.drawMatches(img1,kp1,img2,kp2,newInliers,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img5)
                plt.title("Fundamental+Homography Filter")
                plt.show()
                output_file.write("Image of new inliers consistent with Fundamental and Homography matrix: \n")
                
                #q7 comparing the ratios of inliers found using the fundamental matrix to that of inliers found using homography matrix
                if len(newInliers) < 25 or newInlierRatio < 0.3:
                    output_file.write("Failed threshold for new inliers calculations:\n")
                    output_file.write("Amount of new inliers computed by homography matrix or ratio compared to amount of old inliers computed by fundamental matrix too low\n")
                    continue
                else:
                    output_file.write("Passed threshold for new inliers calculations:\n")
                    output_file.write("Ratio and amount of new inliers computed by homography matrix is above threshold\n")

                #q8 building the mosaic if the new inlier calculations pass my threshold
                mosaic = create_mosaic(img1, img2, H)
                plt.imshow(mosaic, cmap='gray')
                plt.title("Mosaic of Aligned Images")
                plt.show()
                output_file.write("Image of overlapping scenes: \n")
    output_file.close()
    print(f"All outputs have been saved to {arg2}")

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])

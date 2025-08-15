import os
import shutil
import subprocess
import sys
import time
import pdfplumber
import ants
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

working_path = r'/home/neurorad/radiplab/radip6/working/3t-7t-hipposeg-working3'
nii_path = os.path.join(working_path, 'NII')

def preprocess_no_mni_extraction_contrast(reprocess=True):
    nii_path = os.path.join(working_path, 'NII')
    niip_path = os.path.join(working_path, 'NII_preprocessed')

    if not os.path.exists(niip_path):
        os.mkdir(niip_path)
    if reprocess:
        if os.path.exists(niip_path):
            shutil.rmtree(niip_path)
        os.mkdir(niip_path)

    for nii_prid_folder in sorted(os.listdir(nii_path)):
        print("Processing " + nii_prid_folder)
        niip_prid_path = os.path.join(niip_path, nii_prid_folder)
        nii_prid_path = os.path.join(nii_path, nii_prid_folder)
        if not os.path.exists(niip_prid_path):
            os.mkdir(niip_prid_path)
        
        # There is guaranteed to be a 7T T1 and 3T T1
        t1_7t_filename = None
        t1_3t_filename = None
        t1c_7t_filename = None
        t1c_3t_filename = None
        
        for nii_file in os.listdir(nii_prid_path):
            magnetic_strength = nii_file.split('_')[0]
            series_description = nii_file.split('_')[5].split('.')[0]
            if magnetic_strength == '7T':
                if series_description == 'T1':
                    t1_7t_filename = nii_file
                elif series_description == 'T1C': 
                    t1c_7t_filename = nii_file
            elif magnetic_strength == '3T':
                if series_description == 'T1':
                    t1_3t_filename = nii_file
                elif series_description == 'T1C': 
                    t1c_3t_filename = nii_file

        t1_7t_p_filename = t1_7t_filename.split('.')[0] + '_06.nii.gz'
        t1_7t_p_path = os.path.join(niip_prid_path, t1_7t_p_filename)

        t1c_7t_p_filename = None
        t1c_7t_p_path = None
        if t1c_7t_filename is not None:
            t1c_7t_p_filename = t1c_7t_filename.split('.')[0] + '_06.nii.gz'
            t1c_7t_p_path = os.path.join(niip_prid_path, t1c_7t_p_filename)

        t1_3t_p_filename = t1_3t_filename
        t1_3t_p_path = os.path.join(niip_prid_path, t1_3t_p_filename)

        t1_3t_pdr_filename = t1_3t_filename.split('.')[0] + '_dr.nii.gz'
        t1_3t_pdr_path = os.path.join(niip_prid_path, t1_3t_pdr_filename)

        t1c_3t_p_filename = None
        t1c_3t_p_path = None
        if t1c_3t_filename is not None:
            t1c_3t_p_filename = t1c_3t_filename
            t1c_3t_p_path = os.path.join(niip_prid_path, t1c_3t_p_filename)

        # ***** 7T T1
        t1_7t_path = os.path.join(nii_prid_path, t1_7t_filename)
        
        # Bias correction
        t1_7t_b_filename = t1_7t_filename.split('.')[0] + '-b.nii.gz'
        t1_7t_b_path = os.path.join(niip_prid_path, t1_7t_b_filename)
        if not os.path.exists(t1_7t_b_path) and not os.path.exists(t1_7t_p_path):
            t1_7t_re_image = ants.image_read(t1_7t_path)
            t1_7t_b_image = ants.n4_bias_field_correction(t1_7t_re_image)
            ants.image_write(t1_7t_b_image, t1_7t_b_path)
        
        # Normalize signal intensities
        t1_7t_bn_filename = t1_7t_filename.split('.')[0] + '-bn.nii.gz'
        t1_7t_bn_path = os.path.join(niip_prid_path, t1_7t_bn_filename)
        if not os.path.exists(t1_7t_bn_path) and not os.path.exists(t1_7t_p_path):
            image_array = t1_7t_b_image.numpy()
            z_score_normalized = (image_array - np.mean(image_array)) / np.std(image_array) # Normalize
            scaled_image_array = np.interp(z_score_normalized, (z_score_normalized.min(), z_score_normalized.max()), (0, 1000)) # Scale to 0-1000
            t1_7t_bn_image = ants.from_numpy(scaled_image_array)
            t1_7t_bn_image.set_spacing(t1_7t_b_image.spacing)
            t1_7t_bn_image.set_origin(t1_7t_b_image.origin)
            t1_7t_bn_image.set_direction(t1_7t_b_image.direction)
            ants.image_write(t1_7t_bn_image, t1_7t_bn_path)

            # Convert to uint16
            # Convert the image to SimpleITK and then to uint16
            sitk_image = sitk.ReadImage(t1_7t_bn_path)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
            sitk.WriteImage(sitk_image, t1_7t_bn_path)

        # Resample to 0.7, 0.8, 0.9, 1.0 linear and bspline
        t1_7t_07_l_filename = t1_7t_filename.split('.')[0] + '_07l.nii.gz'
        t1_7t_07_l_path = os.path.join(niip_prid_path, t1_7t_07_l_filename)    
        t1_7t_10_l_filename = t1_7t_filename.split('.')[0] + '_10l.nii.gz'
        t1_7t_10_l_path = os.path.join(niip_prid_path, t1_7t_10_l_filename)
        if not os.path.exists(t1_7t_07_l_path):
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_07_l_path,
                                resolution=(0.7,0.7,0.7),
                                interp_type=0)
            t1_7t_07_b_filename = t1_7t_filename.split('.')[0] + '_07b.nii.gz'
            t1_7t_07_b_path = os.path.join(niip_prid_path, t1_7t_07_b_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_07_b_path,
                                resolution=(0.7,0.7,0.7),
                                interp_type=4)
            t1_7t_08_l_filename = t1_7t_filename.split('.')[0] + '_08l.nii.gz'
            t1_7t_08_l_path = os.path.join(niip_prid_path, t1_7t_08_l_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_08_l_path,
                                resolution=(0.8,0.8,0.8),
                                interp_type=0)
            t1_7t_08_b_filename = t1_7t_filename.split('.')[0] + '_08b.nii.gz'
            t1_7t_08_b_path = os.path.join(niip_prid_path, t1_7t_08_b_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_08_b_path,
                                resolution=(0.8,0.8,0.8),
                                interp_type=4)
            t1_7t_09_l_filename = t1_7t_filename.split('.')[0] + '_09l.nii.gz'
            t1_7t_09_l_path = os.path.join(niip_prid_path, t1_7t_09_l_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_09_l_path,
                                resolution=(0.9,0.9,0.9),
                                interp_type=0)
            t1_7t_09_b_filename = t1_7t_filename.split('.')[0] + '_09b.nii.gz'
            t1_7t_09_b_path = os.path.join(niip_prid_path, t1_7t_09_b_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_09_b_path,
                                resolution=(0.9,0.9,0.9),
                                interp_type=4)
            t1_7t_10_l_filename = t1_7t_filename.split('.')[0] + '_10l.nii.gz'
            t1_7t_10_l_path = os.path.join(niip_prid_path, t1_7t_10_l_filename)            
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_10_l_path,
                                resolution=(1.0,1.0,1.0),
                                interp_type=0)
            t1_7t_10_b_filename = t1_7t_filename.split('.')[0] + '_10b.nii.gz'
            t1_7t_10_b_path = os.path.join(niip_prid_path, t1_7t_10_b_filename)
            resample_to_resolution(t1_path=t1_7t_bn_path, 
                                output_path=t1_7t_10_b_path,
                                resolution=(1.0,1.0,1.0),
                                interp_type=4)
            

        # ***** Process 3T T1
        # Register 3T T1 to 7T T1
        t1_3t_r_filename = t1_3t_filename.split('.')[0] + '-r.nii.gz'
        t1_3t_r_path = os.path.join(niip_prid_path, t1_3t_r_filename)        
        if not os.path.exists(t1_3t_r_path) and not os.path.exists(t1_3t_p_path):
            # Resample 7T to 3T resolution first
            t1_7t = ants.image_read(t1_7t_path)
            t1_3t_path = os.path.join(nii_prid_path, t1_3t_filename)
            t1_3t = ants.image_read(t1_3t_path)

            # Resample the 7T image to the resolution of the 3T image
            t1_3t_spacing = t1_3t.spacing  # Get the spatial resolution of the 3T image
            resampled_t1_7t = ants.resample_image(
                t1_7t,
                resample_params=t1_3t_spacing,
                use_voxels=False,  # Indicates spacing in mm
                interp_type=4  # linear interpolation - ok to degrade a little for this temp file
            )

            # Register the 3T T1 image to the resampled 7T T1 image
            registration = ants.registration(
                fixed=resampled_t1_7t,
                moving=t1_3t,
                type_of_transform='Rigid'
            )

            # Apply the registration transform using B-spline interpolation
            registered_t1_3t = ants.apply_transforms(
                fixed=resampled_t1_7t,
                moving=t1_3t,
                transformlist=registration['fwdtransforms'],
                interpolator='bSpline'  # Use B-spline interpolation for smooth results
            )
            ants.image_write(registered_t1_3t, t1_3t_r_path)

        # Bias correction
        t1_3t_rb_filename = t1_3t_filename.split('.')[0] + '-rb.nii.gz'
        t1_3t_rb_path = os.path.join(niip_prid_path, t1_3t_rb_filename)
        if not os.path.exists(t1_3t_rb_path) and not os.path.exists(t1_3t_p_path):
            t1_3t_r_image = ants.image_read(t1_3t_r_path)
            t1_3t_rb_image = ants.n4_bias_field_correction(t1_3t_r_image)
            ants.image_write(t1_3t_rb_image, t1_3t_rb_path)

        # Normalize signal intensities
        t1_3t_rbn_filename = t1_3t_filename.split('.')[0] + '-rbn.nii.gz'
        t1_3t_rbn_path = os.path.join(niip_prid_path, t1_3t_rbn_filename)
        if not os.path.exists(t1_3t_rbn_path) and not os.path.exists(t1_3t_p_path):
            t1_3t_rb_image = ants.image_read(t1_3t_rb_path)
            image_array = t1_3t_rb_image.numpy()
            z_score_normalized = (image_array - np.mean(image_array)) / np.std(image_array) # Normalize
            scaled_image_array = np.interp(z_score_normalized, (z_score_normalized.min(), z_score_normalized.max()), (0, 1000)) # Scale to 0-1000
            t1_3t_rbn_image = ants.from_numpy(scaled_image_array)
            t1_3t_rbn_image.set_spacing(t1_3t_rb_image.spacing)
            t1_3t_rbn_image.set_origin(t1_3t_rb_image.origin)
            t1_3t_rbn_image.set_direction(t1_3t_rb_image.direction)
            ants.image_write(t1_3t_rbn_image, t1_3t_rbn_path)

            # Convert to uint16
            # Convert the image to SimpleITK and then to uint16
            sitk_image = sitk.ReadImage(t1_3t_rbn_path)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
            sitk.WriteImage(sitk_image, t1_3t_rbn_path)

        # Also perform deformable registration of the 3T to 7T 1.0 mm - try to account for distortion
        # Registration
        t1_3t_dr_filename = t1_3t_filename.split('.')[0] + '-dr.nii.gz'
        t1_3t_dr_path = os.path.join(niip_prid_path, t1_3t_dr_filename)        
        if not os.path.exists(t1_3t_dr_path) and not os.path.exists(t1_3t_pdr_path):
            t1_7t = ants.image_read(t1_7t_10_l_path)
            t1_3t = ants.image_read(t1_3t_path)

            # Register the 3T T1 image to the resampled 7T T1 image
            registration = ants.registration(
                fixed=t1_7t,
                moving=t1_3t,
                type_of_transform='SyN'
            )

            # Apply the registration transform using B-spline interpolation
            registered_t1_3t = ants.apply_transforms(
                fixed=t1_7t,
                moving=t1_3t,
                transformlist=registration['fwdtransforms'],
                interpolator='bSpline'  # Use B-spline interpolation for smooth results
            )
            ants.image_write(registered_t1_3t, t1_3t_dr_path)
        
        # Bias correction
        t1_3t_drb_filename = t1_3t_filename.split('.')[0] + '-drb.nii.gz'
        t1_3t_drb_path = os.path.join(niip_prid_path, t1_3t_drb_filename)
        if not os.path.exists(t1_3t_drb_path) and not os.path.exists(t1_3t_pdr_path):
            t1_3t_dr_image = ants.image_read(t1_3t_dr_path)
            t1_3t_drb_image = ants.n4_bias_field_correction(t1_3t_dr_image)
            ants.image_write(t1_3t_drb_image, t1_3t_drb_path)

        # Normalize signal intensities
        t1_3t_drbn_filename = t1_3t_filename.split('.')[0] + '-drbn.nii.gz'
        t1_3t_drbn_path = os.path.join(niip_prid_path, t1_3t_drbn_filename)
        if not os.path.exists(t1_3t_drbn_path) and not os.path.exists(t1_3t_pdr_path):
            t1_3t_ber_image = ants.image_read(t1_3t_drb_path)
            image_array = t1_3t_ber_image.numpy()
            z_score_normalized = (image_array - np.mean(image_array)) / np.std(image_array) # Normalize
            scaled_image_array = np.interp(z_score_normalized, (z_score_normalized.min(), z_score_normalized.max()), (0, 1000)) # Scale to 0-1000
            t1_3t_drbn_image = ants.from_numpy(scaled_image_array)
            t1_3t_drbn_image.set_spacing(t1_3t_ber_image.spacing)
            t1_3t_drbn_image.set_origin(t1_3t_ber_image.origin)
            t1_3t_drbn_image.set_direction(t1_3t_ber_image.direction)
            ants.image_write(t1_3t_drbn_image, t1_3t_drbn_path)

            # Convert to uint16
            # Convert the image to SimpleITK and then to uint16
            sitk_image = sitk.ReadImage(t1_3t_drbn_path)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
            sitk.WriteImage(sitk_image, t1_3t_drbn_path)

        # Delete intermediaries and rename finals
        if not os.path.exists(t1_7t_p_path):
            os.remove(t1_7t_b_path)
            os.rename(t1_7t_bn_path, t1_7t_p_path)

        if not os.path.exists(t1_3t_p_path):
            os.remove(t1_3t_r_path)
            os.remove(t1_3t_rb_path)
            os.rename(t1_3t_rbn_path, t1_3t_p_path)

        if not os.path.exists(t1_3t_pdr_path):
            os.remove(t1_3t_dr_path)
            os.remove(t1_3t_drb_path)
            os.rename(t1_3t_drbn_path, t1_3t_pdr_path)

def resample_to_resolution(t1_path, output_path, resolution=(0.07, 0.07, 0.07), interp_type=0):
    """
    Resample a T1 image to the specified spatial resolution.

    Parameters:
    - t1_path: Path to the input T1-weighted MRI image.
    - output_path: Path to save the resampled image.
    - resolution: Tuple specifying the target spatial resolution (default is 0.07 mm isotropic).
    - interp_type: Interpolation type for resampling (default is 1 for linear interpolation).
        Options:
        - 0: Linear
        - 1: Nearest neighbor
        - 2: Gaussian
        - 3: Windowed sinc
        - 4: Bspline

    Returns:
    - Path to the resampled T1 image.
    """
    # Load the T1 image
    t1_image = ants.image_read(t1_path)

    # Resample the image to the specified resolution
    resampled_image = ants.resample_image(
        image=t1_image,
        resample_params=resolution,
        use_voxels=False,  # Indicates that resample_params specifies spacing in mm
        interp_type=interp_type  # Interpolation type
    )

    # Save the resampled image
    ants.image_write(resampled_image, output_path)

    # Convert to uint16
    # Convert the image to SimpleITK and then to uint16
    sitk_image = sitk.ReadImage(output_path)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    sitk.WriteImage(sitk_image, output_path)

    return output_path


def segment_e2dhipseg(reprocess=False):
    niip_path = os.path.join(working_path, 'NII_preprocessed')
    total_studies = len(os.listdir(niip_path))
    total_time = 0

    print(f"segment_e2dhipseg processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(niip_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(niip_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(niip_path, prid_folder, file)
                    if 'e2dhipseg' in file and file != 'e2dhipseg_masks':
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Define paths
            t1_files = []
            for file in os.listdir(study_path):
                #if not 't1c' in file.lower() and not 'label' in file.lower() and '3t' in file.lower():
                if not 't1c' in file.lower() and not 'label' in file.lower() and not 'prob' in file.lower():
                    t1_files.append(file)

            for t1_file in t1_files:
                #if 'T1_dr.nii.gz' in t1_file:
                if True:
                    t1_filename = t1_file.split('.')[0]
                    # Define the paths
                    r_hippo_prob_path = os.path.join(study_path, t1_filename + '-r-hippocampus-e2dhipseg-prob.nii.gz')
                    l_hippo_prob_path = os.path.join(study_path, t1_filename + '-l-hippocampus-e2dhipseg-prob.nii.gz')

                    if not os.path.exists(r_hippo_prob_path) or not os.path.exists(l_hippo_prob_path):
                        e2d_result_path = os.path.join(study_path, 'e2dhipseg_masks')
                        if not os.path.exists(e2d_result_path):
                            # Define the paths
                            # Convert to RAS
                            t1_path = os.path.join(study_path, t1_file)
                            output_path = os.path.join(os.path.dirname(t1_path), 'T1_for_e2dhipseg.nii.gz')
                            command = f"mri_convert --out_orientation RAS {t1_path} {output_path}"
                            process = subprocess.run(command, shell=True, executable='/bin/bash')

                            python_executable = "/home/neurorad/anaconda3/envs/radiology/bin/python3"
                            e2dhipseg_path = "/home/neurorad/radiplab/e2dhipseg"
                            script_path = os.path.join(e2dhipseg_path, "run.py")
                            command = [python_executable, script_path, output_path, '-reg']
                            result = subprocess.run(command, cwd=e2dhipseg_path) # output is in ./e2dhipseg_masks

                            os.remove(output_path)
                        
                        e2d_result_file = None
                        for e2d_file in os.listdir(e2d_result_path):
                            if 'e2dhipmask' in e2d_file:
                                e2d_result_file = e2d_file
                        e2d_result_file_path = os.path.join(e2d_result_path, e2d_result_file)
                        e2d_sitk = sitk.ReadImage(e2d_result_file_path)

                        if np.max(sitk.GetArrayFromImage(e2d_sitk)) > 0:
                            # Now divide into right and left hippocampus
                            # Step 1: Threshold the probability image for anything above 0
                            thresholded = sitk.BinaryThreshold(e2d_sitk, lowerThreshold=1e-6, upperThreshold=1.0, insideValue=1, outsideValue=0)

                            # Step 2: Perform connected component analysis to label the two largest components
                            connected_components = sitk.ConnectedComponent(thresholded)

                            # Step 3: Relabel the connected components to label them by size
                            relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

                            # Step 4: Use LabelShapeStatistics to get centroids of the two largest components
                            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
                            label_shape_filter.Execute(relabeled_components)

                            if label_shape_filter.HasLabel(1) and label_shape_filter.HasLabel(2): # There should be 2 components
                                # Get centroids of the two largest components
                                centroid_1 = label_shape_filter.GetCentroid(1)
                                centroid_2 = label_shape_filter.GetCentroid(2)

                                # Step 5: Reverse the left-right assignment by comparing centroids based on the x-coordinate
                                l_mask = None
                                r_mask = None
                                if centroid_1[0] > centroid_2[0]:
                                    # Component 1 is on the right, Component 2 is on the left
                                    r_mask = sitk.BinaryThreshold(relabeled_components, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)
                                    l_mask = sitk.BinaryThreshold(relabeled_components, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
                                else:
                                    # Component 1 is on the left, Component 2 is on the right
                                    r_mask = sitk.BinaryThreshold(relabeled_components, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
                                    l_mask = sitk.BinaryThreshold(relabeled_components, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)

                                # Step 6: Multiply the original probability image by the masks to retain all probability values for each component
                                l_e2d_sitk = e2d_sitk * sitk.Cast(l_mask, e2d_sitk.GetPixelID())
                                r_e2d_sitk = e2d_sitk * sitk.Cast(r_mask, e2d_sitk.GetPixelID())

                                sitk.WriteImage(r_e2d_sitk, r_hippo_prob_path)
                                sitk.WriteImage(l_e2d_sitk, l_hippo_prob_path)

                                # Now save 11 labels at different threshold levels
                                #thresholds = [1e-6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                                thresholds = [0.3]
                                
                                # Loop over each threshold and create/save the corresponding label image
                                for threshold in thresholds:
                                    # Create a binary mask for the current threshold
                                    r_thresholded_label = sitk.BinaryThreshold(r_e2d_sitk, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                                    l_thresholded_label = sitk.BinaryThreshold(l_e2d_sitk, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                                    
                                    # Define the file name using the current threshold
                                    r_label_filename = t1_filename + '-r-hippocampus-e2dhipseg' + str(threshold) + '-label.nii.gz'
                                    r_label_path = os.path.join(study_path, r_label_filename)
                                    l_label_filename = t1_filename + '-l-hippocampus-e2dhipseg' + str(threshold) + '-label.nii.gz'
                                    l_label_path = os.path.join(study_path, l_label_filename)

                                    # Save the label image
                                    sitk.WriteImage(r_thresholded_label, r_label_path)
                                    sitk.WriteImage(l_thresholded_label, l_label_path)

                        shutil.rmtree(e2d_result_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"e2dhipseg processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    


def segment_hippodeep(reprocess=False):
    niip_path = os.path.join(working_path, 'NII_preprocessed')
    total_studies = len(os.listdir(niip_path))
    total_time = 0

    print(f"hippodeep processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(niip_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(niip_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path): # Always reprocess
                    file_path = os.path.join(niip_path, prid_folder, file)
                    if 'hippodeep' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Just isolate the hippocampus label
            # Define paths
            t1_files = []
            for file in os.listdir(study_path):
                if not 't1c' in file.lower() and 'label' not in file.lower() and not 'prob' in file.lower() and os.path.isfile(os.path.join(study_path, file)):
                    t1_files.append(file)

            for t1_file in t1_files:
                t1_filename = t1_file.split('.')[0]
                # Define the paths
                r_hippo_prob_path = os.path.join(study_path, t1_filename + '-r-hippocampus-hippodeep-prob.nii.gz')
                l_hippo_prob_path = os.path.join(study_path, t1_filename + '-l-hippocampus-hippodeep-prob.nii.gz')
                if not os.path.exists(r_hippo_prob_path) or not os.path.exists(l_hippo_prob_path):
                    script_path = "/home/neurorad/radiplab/hippodeep_pytorch/deepseg1.sh"
                    t1_path = os.path.join(study_path, t1_file)
                    output_path = os.path.join('/home/neurorad/Downloads/hippodeep')
                    if os.path.exists(output_path):
                        shutil.rmtree(output_path)
                    os.mkdir(output_path)
                    input_image = os.path.join(output_path, 'T1.nii.gz')
                    shutil.copy2(t1_path, input_image)

                    # Run the shell script with the input image and specify the working directory for output
                    command = ['bash', script_path, input_image]
                    result = subprocess.run(command, cwd=output_path)

                    # Loop over each threshold and create/save the corresponding label image
                    r_prob_mask_path = os.path.join(output_path, 'T1_mask_R.nii.gz')
                    r_prob_mask_image = sitk.ReadImage(r_prob_mask_path)
                    r_prob_mask_array = sitk.GetArrayFromImage(r_prob_mask_image)
                    r_prob_mask_array_rescaled = np.interp(r_prob_mask_array, (np.min(r_prob_mask_array), np.max(r_prob_mask_array)), (0, 1))
                    r_prob_mask_image_rescaled = sitk.GetImageFromArray(r_prob_mask_array_rescaled)
                    r_prob_mask_image_rescaled.CopyInformation(r_prob_mask_image)
                    sitk.WriteImage(r_prob_mask_image_rescaled, r_hippo_prob_path)

                    l_prob_mask_path = os.path.join(output_path, 'T1_mask_L.nii.gz')
                    l_prob_mask_image = sitk.ReadImage(l_prob_mask_path)
                    l_prob_mask_array = sitk.GetArrayFromImage(l_prob_mask_image)
                    l_prob_mask_array = l_prob_mask_array.astype(np.float32)
                    l_prob_mask_array_rescaled = np.interp(l_prob_mask_array, (np.min(l_prob_mask_array), np.max(l_prob_mask_array)), (0, 1))
                    l_prob_mask_image_rescaled = sitk.GetImageFromArray(l_prob_mask_array_rescaled)
                    l_prob_mask_image_rescaled.CopyInformation(l_prob_mask_image)
                    sitk.WriteImage(l_prob_mask_image_rescaled, l_hippo_prob_path)

                    #thresholds = [1e-6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                    thresholds = [0.3]
                    for threshold in thresholds: # Loop over each threshold and create/save the corresponding label image
                        # Create a binary mask for the current threshold
                        r_thresholded_label = sitk.BinaryThreshold(r_prob_mask_image_rescaled, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                        l_thresholded_label = sitk.BinaryThreshold(l_prob_mask_image_rescaled, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                        
                        # Define the file name using the current threshold
                        r_label_filename = t1_filename + '-r-hippocampus-hippodeep' + str(threshold) + '-label.nii.gz'
                        r_label_path = os.path.join(study_path, r_label_filename)
                        l_label_filename = t1_filename + '-l-hippocampus-hippodeep' + str(threshold) + '-label.nii.gz'
                        l_label_path = os.path.join(study_path, l_label_filename)

                        # Save the label image
                        sitk.WriteImage(r_thresholded_label, r_label_path)
                        sitk.WriteImage(l_thresholded_label, l_label_path)

                    shutil.rmtree(output_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"hippodeep processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    


def segment_HippMapp3r(reprocess=False):
    niip_path = os.path.join(working_path, 'NII_preprocessed')
    total_studies = len(os.listdir(niip_path))
    total_time = 0

    print(f"segment_hippmapper processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(niip_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(niip_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(niip_path, prid_folder, file)
                    if 'hippmapper' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            t1_files = []
            for file in os.listdir(study_path):
                if not 't1c' in file.lower() and not 'label' in file.lower() and not 'prob' in file.lower() and not 'aparc' in file.lower() and os.path.isfile(os.path.join(study_path, file)):
                    t1_files.append(file)

            for t1_file in t1_files:
                t1_filename = t1_file.split('.')[0]

                # Define paths
                r_hippo_path = os.path.join(study_path, t1_filename + '-r-hippocampus-hippmapper-label.nii.gz')
                l_hippo_path = os.path.join(study_path, t1_filename + '-l-hippocampus-hippmapper-label.nii.gz')
                
                if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                    hippmapper_working_path = os.path.join('/home/neurorad/Downloads/hippmapper_working')
                    if os.path.exists(hippmapper_working_path):
                        shutil.rmtree(hippmapper_working_path)

                    t1_path = os.path.join(hippmapper_working_path, t1_file)

                    # Example command: hippmapper seg_hipp --t1w /home/neurorad/radiplab/radip6/working/hippocampal-seg-working/nii/00000001/hippmapper_working/T1.nii.gz
                    os.mkdir(hippmapper_working_path)
                    shutil.copy2(os.path.join(study_path, t1_file), t1_path)

                    # Reformat T1
                    # Convert the skull-stripped image to RPI orientation using FreeSurfer's mri_convert
                    # Bizarre...hippmapper wants LPI or RPI, and when I mri_convert, hippmapper interprets it as the opposite
                    # So RAS with mri_convert will be LPI for hippmapper, and then it works
                    output_path_rpi = os.path.join(hippmapper_working_path, 't1_skull_stripped_RAS.nii.gz')
                    command = f"mri_convert --out_orientation RAS {t1_path} {output_path_rpi}"
                    process = subprocess.run(command, shell=True, executable='/bin/bash')
                    
                    # Step 7: Segment with hippmapper
                    command = f"""
                    source /home/neurorad/anaconda3/bin/activate hippmapper && \
                    hippmapper seg_hipp --t1w {output_path_rpi}
                    """
                    # Run hippmapper in a subprocess
                    process = subprocess.run(command, shell=True, executable='/bin/bash')
                        
                    # Step 8: Save results
                    # Result file is t1_skull_stripped_RAS_T1acq_hipp_pred.nii.gz - right hippo = 1, left hippo = 2
                    label_map_path = os.path.join(hippmapper_working_path, 't1_skull_stripped_RAS_T1acq_hipp_pred.nii.gz')

                    # Convert back to T1 orientation
                    label_map_converted_path = os.path.join(hippmapper_working_path, 'predictions-converted.nii.gz')                
                    subprocess.call([
                        r'/usr/local/freesurfer/7.3.2/bin/mri_convert',
                        '--reslice_like', t1_path,  # Match size and voxel grid of the T1 template
                        '--resample_type', 'nearest',  # Use nearest-neighbor interpolation to preserve mask values
                        label_map_path,  # Input segmentation file
                        label_map_converted_path  # Output file
                    ]) 

                    label_map_image = sitk.ReadImage(label_map_converted_path)

                    # Step 1: Generate the right hippocampus label map (label = 1)
                    right_hippo_image = sitk.BinaryThreshold(label_map_image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)

                    # Step 2: Generate the left hippocampus label map (label = 2)
                    left_hippo_image = sitk.BinaryThreshold(label_map_image, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)

                    # Step 3: Save the right and left hippocampus label maps
                    sitk.WriteImage(right_hippo_image, r_hippo_path)
                    sitk.WriteImage(left_hippo_image, l_hippo_path)

                    shutil.rmtree(hippmapper_working_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"hippmapper processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    


def segment_fastsurfer(reprocess=False):
    niip_path = os.path.join(working_path, 'NII_preprocessed')
    total_studies = len(os.listdir(niip_path))
    total_time = 0    
    print(f"Processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(niip_path))):
        study_path = os.path.join(niip_path, prid_folder)
        start_time = time.time()
        if reprocess:
            for file in os.listdir(study_path):
                file_path = os.path.join(niip_path, prid_folder, file)
                if 'aparc' in file or 'fastsurfer' in file:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

        seg_tmp_path = '/home/neurorad/Downloads/fastsurfer'
        # Clean up tmp path
        if os.path.exists(seg_tmp_path):
            shutil.rmtree(seg_tmp_path)

        t1_files = []
        for file in os.listdir(study_path):
            if not 't1c' in file.lower() and not 'label' in file.lower() and not 'prob' in file.lower():
                t1_files.append(file)

        for t1_file in t1_files:
            t1_filename = t1_file.split('.')[0]

            # Define paths
            r_hippo_path = os.path.join(study_path, t1_filename + '-r-hippocampus-fastsurfer-label.nii.gz')
            l_hippo_path = os.path.join(study_path, t1_filename + '-l-hippocampus-fastsurfer-label.nii.gz')
            asegdkt_segfile_nii_path = os.path.join(study_path, t1_filename + '.aparc.DKTatlas+aseg.deep.nii.gz')
            study_path = os.path.join(niip_path, prid_folder)
            if not os.path.exists(r_hippo_path):
                t1_path = os.path.join(study_path, t1_file)
                os.mkdir(seg_tmp_path)
                
                # Quick segmentation
                subject_id = prid_folder.split('-')[0]
                asegdkt_segfile_path = os.path.join(seg_tmp_path, subject_id, 'aparc.DKTatlas+aseg.deep.mgz')
                conformed_path = os.path.join(seg_tmp_path, subject_id, 'conformed.mgz')
                subprocess.run(['/home/neurorad/radiplab/FastSurfer/run_fastsurfer.sh',
                                    '--t1', 
                                    t1_path, 
                                    '--sid', 
                                    subject_id, 
                                    '--sd', 
                                    seg_tmp_path, 
                                    '--asegdkt_segfile',
                                    asegdkt_segfile_path,
                                    '--conformed_name',
                                    conformed_path,
                                    '--parallel', 
                                    '--threads', 
                                    '4',
                                    '--seg_only',
                                    '--no_cereb',
                                    '--no_biasfield'],
                                check=True)

                # Convert to .nii.gz and copy
                if not os.path.exists(asegdkt_segfile_nii_path):
                    #subprocess.call([r'/usr/local/freesurfer/7.3.2/bin/mri_convert', asegdkt_segfile_path, asegdkt_segfile_nii_path])
                    
                    subprocess.call([
                        r'/usr/local/freesurfer/7.3.2/bin/mri_convert',
                        '--reslice_like', t1_path,  # Match size and voxel grid of the T1 template
                        '--resample_type', 'nearest',  # Use nearest-neighbor interpolation to preserve mask values
                        asegdkt_segfile_path,  # Input segmentation file
                        asegdkt_segfile_nii_path  # Output file
                    ]) 

                shutil.rmtree(seg_tmp_path)

                # Just isolate the hippocampus label                
                # Isolate hippocampi from wmparc.DKTatlas.mapped.nii.gz
                # 17 = left hippocampus
                # 53 = right hippocampus            
                seg = sitk.ReadImage(asegdkt_segfile_nii_path)
                seg_data = sitk.GetArrayFromImage(seg)            
                r_hippo_data = np.zeros_like(seg_data)
                r_hippo_indices = np.where(seg_data == 53)
                r_hippo_data[r_hippo_indices] = 1
                r_hippo = sitk.GetImageFromArray(r_hippo_data)
                r_hippo.CopyInformation(seg)

                l_hippo_data = np.zeros_like(seg_data)
                l_hippo_indices = np.where(seg_data == 17)
                l_hippo_data[l_hippo_indices] = 1
                l_hippo = sitk.GetImageFromArray(l_hippo_data)
                l_hippo.CopyInformation(seg)

                sitk.WriteImage(r_hippo, r_hippo_path)
                sitk.WriteImage(l_hippo, l_hippo_path)
                os.remove(asegdkt_segfile_nii_path)

        end_time = time.time()
        elapsed_time = end_time - start_time  # Time taken for one study
        total_time += elapsed_time  # Total time so far
        avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
        remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
        estimated_time_left = avg_time_per_study * remaining_studies
        print(f"Processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")

def resample_image():
    image_folder = r'/home/neurorad/radiplab/radip6/working/3t-7t-hipposeg-working3/NII_preprocessed/00000015'
    image_filename = '3T_00000015_00000029_2_20250218_T1.nii.gz'
    image_path = os.path.join(image_folder, image_filename)

    # --- read the image ---
    img = sitk.ReadImage(image_path)

    # --- target spacing and derived size for 1 mm isovoxel ---
    new_spacing = (1.0, 1.0, 1.0)
    #new_spacing = (0.6, 0.6, 0.6)
    old_spacing = img.GetSpacing()
    old_size    = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]

    # --- resample (cubic for intensities; switch to sitkNearestNeighbor for labels) ---
    resampled_img = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkBSpline,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0,
        img.GetPixelID()
    )

    # --- save alongside the original ---
    sitk.WriteImage(resampled_img, image_path)


def segment_AssemblyNet(reprocess=False):
    niip_path = os.path.join(working_path, 'NII_preprocessed')
    total_studies = len(os.listdir(niip_path))
    total_time = 0

    print(f"segment_AssemblyNet processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(niip_path))):
        if int(prid_folder) < 38:
            start_time = time.time()
            study_path = os.path.join(niip_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(niip_path, prid_folder, file)
                    if 'assemblynet' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            t1_files = []
            for file in os.listdir(study_path):
                if not 't1c' in file.lower() and 'label' not in file.lower() and not 'prob' in file.lower():
                    t1_files.append(file)

            for t1_file in t1_files:
                t1_filename = t1_file.split('.')[0]

                # Define paths
                r_hippo_path = os.path.join(study_path, t1_filename + '-r-hippocampus-assemblynet-label.nii.gz')
                l_hippo_path = os.path.join(study_path, t1_filename + '-l-hippocampus-assemblynet-label.nii.gz')
                
                if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                    tmp_path = os.path.join('/home/neurorad/Downloads/assemblynet_working')
                    if os.path.exists(tmp_path):
                        shutil.rmtree(tmp_path)
                    os.mkdir(tmp_path)

                    t1_path = os.path.join(tmp_path, 'T1.nii.gz')
                    shutil.copy2(os.path.join(study_path, t1_file), t1_path)
                    docker_command = [
                        "sudo", "docker", "run", "--rm", "--gpus", '"device=0"',
                        "-v", f"{tmp_path}:/data", "volbrain/assemblynet:1.0.0", f"/data/T1.nii.gz"
                    ]

                    sudo_password = "2018mlres!"
                    result = subprocess.run(
                        ["sudo", "-S"] + docker_command,
                        input=sudo_password + "\n",  # Pass the password to sudo
                        text=True,  # Ensures input/output are treated as text
                        check=True,  # Raise exception on failure
                        stdout=sys.stdout,  # Stream stdout to console in real-time
                        stderr=sys.stderr   # Stream stderr to console in real-time
                    )

                    # Save whole brain (native_structures_T1.nii.gz) as assemblynet-labels.nii.gz
                    assemblynet_labels_path = os.path.join(study_path, 'assemblynet-labels.nii.gz')
                    shutil.copy2(os.path.join(tmp_path, 'native_structures_T1.nii.gz'), assemblynet_labels_path)

                    # Extract hippocampi: R = 47, L = 48
                    seg = sitk.ReadImage(assemblynet_labels_path)
                    seg_data = sitk.GetArrayFromImage(seg)            
                    r_hippo_data = np.zeros_like(seg_data)
                    r_hippo_indices = np.where(seg_data == 47)
                    r_hippo_data[r_hippo_indices] = 1
                    r_hippo = sitk.GetImageFromArray(r_hippo_data)
                    r_hippo.CopyInformation(seg)

                    l_hippo_data = np.zeros_like(seg_data)
                    l_hippo_indices = np.where(seg_data == 48)
                    l_hippo_data[l_hippo_indices] = 1
                    l_hippo = sitk.GetImageFromArray(l_hippo_data)
                    l_hippo.CopyInformation(seg)

                    sitk.WriteImage(r_hippo, r_hippo_path)
                    sitk.WriteImage(l_hippo, l_hippo_path)

                    shutil.rmtree(tmp_path)
                    
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"AssemblyNet processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    

def segment_neuroquant():
    dicom_anon_path = r'/media/neurorad/Extreme Pro/working/3t-7t-hipposeg-working3/DICOM-anon-python'
    aetitle = ""
    aec = ""
    host = ""
    port = ""

    for prid_folder in os.listdir(dicom_anon_path):
        prid_path = os.path.join(dicom_anon_path, prid_folder)
        if not os.path.isdir(prid_path):
            continue
        for erid_folder in os.listdir(prid_path):
            erid_path = os.path.join(prid_path, erid_folder)
            if not os.path.isdir(erid_path):
                continue

            process = False
            if int(prid_folder) == 12 and int(erid_folder) == 24:
                process = True
            if int(prid_folder) == 45 and int(erid_folder) == 90:
                process = True
            if process:
                print(f"Sending DICOM from {erid_path}...")
                try:
                    result = subprocess.run([
                        "dcmsend", host, port, erid_path,
                        "-aet", aetitle,
                        "-aec", aec,
                        "+sd",  # send subdirectories
                        "-v"    # verbose
                    ], capture_output=True, text=True, check=True)

                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to send DICOM from {erid_path}")
                    print(f"Error: {e.stderr}")

def identify_outlier_cases():
    volumes_csv = os.path.join(working_path, 'volumes.csv')
    output_csv = os.path.join(working_path, 'outlier_cases.csv')

    df = pd.read_csv(volumes_csv)

    # Melt into long format: one row per PRID, ERID (if available), Side, Algorithm
    long_data = []

    for col in df.columns:
        if not col.startswith(('3T', '7T')):
            continue
        try:
            parts = col.split('-')
            magnet = parts[0]
            resolution = parts[1]
            side = parts[2]
            region = parts[3]
            algorithm = "-".join(parts[4:])
            if algorithm.lower().startswith(('fastsurfer', 'hippmapper')):
                continue  # Skip these algorithms
            for idx, value in df[col].items():
                folder = df.loc[idx, 'Folder']
                long_data.append({
                    'Folder': folder,
                    'Magnet': magnet,
                    'Side': side,
                    'Algorithm': algorithm,
                    'Volume': value
                })
        except Exception as e:
            print(f"Skipping column: {col} due to error: {e}")

    long_df = pd.DataFrame(long_data)

    # Group by Folder + Magnet + Side and analyze volumes across algorithms
    outliers = []

    for (folder, magnet, side), group in long_df.groupby(['Folder', 'Magnet', 'Side']):
        vols = group['Volume']
        mean = vols.mean()
        std = vols.std()
        if std == 0 or np.isnan(std):
            continue  # skip trivial or empty sets
        for _, row in group.iterrows():
            deviation = abs(row['Volume'] - mean)
            if deviation > 2 * std:
                outliers.append({
                    'Folder': folder,
                    'Magnet': magnet,
                    'Side': side,
                    'Algorithm': row['Algorithm'],
                    'Volume': row['Volume'],
                    'Mean (all algs)': mean,
                    'Std Dev': std,
                    'Deviation': deviation
                })

    outlier_df = pd.DataFrame(outliers)
    outlier_df.sort_values(by='Deviation', ascending=False, inplace=True)
    outlier_df.to_csv(output_csv, index=False)
    print(f"✓ Outlier cases saved to {output_csv}")


def process_volumes():
    nii_path = os.path.join(working_path, 'NII_preprocessed')

    # Initialize an empty DataFrame to store results
    results = []

    # Traverse through each folder and process .nii.gz files
    for folder in sorted(os.listdir(nii_path)):
        folder_path = os.path.join(nii_path, folder)
        if os.path.isdir(folder_path):
            print("Processing " + folder)
            # Dictionary to store volumes for this folder
            folder_data = {"Folder": folder}
            for file in os.listdir(folder_path):
                if file.endswith('.nii.gz') and 'label.nii.gz' in file:
                    file_path = os.path.join(folder_path, file)
                    column_name = extract_column_name(file)
                    volume = calculate_volume(file_path)
                    folder_data[column_name] = volume
            results.append(folder_data)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(working_path, 'volumes.csv')
    df.to_csv(output_csv_path, index=False)

def add_neuroquant_volumes():
    pdf_dir = os.path.join(working_path, 'neuroquant_results_pdf')
    orig_volumes_csv = os.path.join(working_path, 'volumes.csv')
    nq_volumes_csv = os.path.join(working_path, 'volumes_nq.csv')
    nq_results_csv = os.path.join(working_path, 'nq_results.csv')

    # Load the original volumes CSV
    volumes_df = pd.read_csv(orig_volumes_csv)

    # Add new columns for NeuroQuant if not already present
    nq_cols = [
        "7T-06-R-Hippocampus-NeuroQuant",
        "7T-06-L-Hippocampus-NeuroQuant",
        "3T-10-R-Hippocampus-NeuroQuant",
        "3T-10-L-Hippocampus-NeuroQuant",
    ]
    for col in nq_cols:
        if col not in volumes_df.columns:
            volumes_df[col] = pd.NA

    # Results table for nq_results.csv
    nq_results = []

    # Loop over PDFs
    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        # Parse PRID, ERID, field strength
        try:
            base = os.path.basename(filename).split('_')[0]
            prid = int(base[:2])
            erid = int(base[2:4])
            field_strength = int(base[4])
        except:
            print(f"Skipping {filename}: couldn't parse PRID/ERID/Field strength")
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue

        # Extract data
        left_volume = left_percentile = right_volume = right_percentile = None
        for line in text.splitlines():
            parts = line.strip().split()
            if line.startswith("Left "):
                try:
                    left_volume = float(parts[1])
                    left_percentile = int(parts[-1])
                except:
                    continue
            elif line.startswith("Right "):
                try:
                    right_volume = float(parts[1])
                    right_percentile = int(parts[-1])
                except:
                    continue

        if None in (left_volume, left_percentile, right_volume, right_percentile):
            print(f"Missing data in {filename}")
            continue

        # Add to nq_results
        nq_results.append({
            "PRID": prid,
            "ERID": erid,
            "Field Strength": field_strength,
            "Left Volume": left_volume,
            "Left Percentile": left_percentile,
            "Right Volume": right_volume,
            "Right Percentile": right_percentile,
        })

        # Add to volumes_nq
        row = volumes_df[volumes_df['Folder'] == prid]
        if row.empty:
            print(f"PRID {prid} not found in volumes.csv")
            continue

        if field_strength == 7:
            volumes_df.loc[volumes_df['Folder'] == prid, "7T-06-R-Hippocampus-NeuroQuant"] = right_volume
            volumes_df.loc[volumes_df['Folder'] == prid, "7T-06-L-Hippocampus-NeuroQuant"] = left_volume
        elif field_strength == 3:
            volumes_df.loc[volumes_df['Folder'] == prid, "3T-10-R-Hippocampus-NeuroQuant"] = right_volume
            volumes_df.loc[volumes_df['Folder'] == prid, "3T-10-L-Hippocampus-NeuroQuant"] = left_volume

    # Save updated volumes.csv
    volumes_df.to_csv(nq_volumes_csv, index=False)
    print(f"✓ Saved updated volumes to {nq_volumes_csv}")

    # Save results table
    nq_df = pd.DataFrame(nq_results)
    nq_df.sort_values(by=["PRID", "Field Strength"], inplace=True)
    nq_df.to_csv(nq_results_csv, index=False)
    print(f"✓ Saved detailed results to {nq_results_csv}")

# Function to calculate volume from a binary label file
def calculate_volume(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    voxel_volume = abs(nib.affines.voxel_sizes(img.affine).prod()) / 1000  # Convert to mL
    return (data > 0).sum() * voxel_volume

# Function to extract column name from filename
def extract_column_name(filename):
    # 7T filename: 7T_00000001_00000002_2_20241101_T1_06-l-hippocampus-fastsurfer-label.nii.gz
    # 3T filename: 3T_00000001_00000001_2_20240716_T1-l-hippocampus-fastsurfer-label.nii.gz
    # 3T filename: 3T_00000001_00000001_2_20240716_T1_dr-r-hippocampus-fastsurfer-label.nii.gz
    # Example filename: 7T-07l-L Hippocampus-Fastsurfer
    # Example filename: 3T-10-L Hippocampus-Fastsurfer
    # Example filename: 3T-10dr-L Hippocampus-Fastsurfer
    parts = filename.split('_')
    magnet_strength = parts[0]
    resolution = None
    second_section = None
    if magnet_strength == '3T':
        if 'T1_dr' not in filename:
            resolution = '10'
            second_section = parts[5]
        else:
            resolution = '10dr'
            second_section = parts[6]
    else:
        second_section = parts[6]
        resolution = second_section.split('-')[0].upper()
    side = second_section.split('-')[1].upper()  # Side (L or R)
    region = second_section.split('-')[2].capitalize()  # Hippocampus
    algorithm = second_section.split('-')[3].capitalize()  # Segmentation algorithm
    return f"{magnet_strength}-{resolution}-{side}-{region}-{algorithm}"


def process_mean_volumes():
    # Load the CSV file
    csv_path = os.path.join(working_path, 'volumes.csv')
    df = pd.read_csv(csv_path)

    # Prepare lists to store labels and mean volumes
    labels = []
    mean_volumes = []

    # Iterate over columns and calculate mean volumes for paired left and right
    for column in df.columns:
        if "-L-Hippocampus" in column:
            # Identify the matching right column
            right_column = column.replace("-L-Hippocampus", "-R-Hippocampus")
            if right_column in df.columns:
                # Calculate mean volume
                mean_volume = (df[column].mean() + df[right_column].mean()) / 2
                labels.append(column.replace("-L-Hippocampus", "-Mean-Hippocampus"))
                mean_volumes.append(mean_volume)

    # Create a new DataFrame with the labels and mean volumes
    mean_df = pd.DataFrame({"Label": labels, "Mean Volume (mL)": mean_volumes})

    # Save the mean volumes to a new CSV file
    output_csv_path = f"{working_path}/mean_volumes.csv"
    mean_df.to_csv(output_csv_path, index=False)
    print(f"Mean volumes saved to {output_csv_path}")


def process_volumes_stats_no_outliers():
    import os
    import pandas as pd
    from scipy.stats import shapiro, ttest_rel, wilcoxon

    volumes_csv = os.path.join(working_path, 'volumes.csv')
    stats_output_csv = os.path.join(working_path, 'stats_volumes_no_outliers.csv')
    outliers_output_csv = os.path.join(working_path, 'stats_volumes_outliers.csv')

    df = pd.read_csv(volumes_csv)

    # Discover algorithms
    algorithms = set()
    for col in df.columns:
        if col.startswith('3T-10-R-Hippocampus-'):
            alg = col.replace('3T-10-R-Hippocampus-', '')
            if f'7T-06-R-Hippocampus-{alg}' in df.columns and f'7T-06-L-Hippocampus-{alg}' in df.columns:
                algorithms.add(alg)

    all_outlier_subjects = set()
    outliers_rows = []
    # --- Pass 1: Find outlier subjects across all algorithms ---
    for alg in sorted(algorithms):
        cols = {
            '3T_L':   f'3T-10-L-Hippocampus-{alg}',
            '3T_R':   f'3T-10-R-Hippocampus-{alg}',
            '3Tdr_L': f'3T-10dr-L-Hippocampus-{alg}',
            '3Tdr_R': f'3T-10dr-R-Hippocampus-{alg}',
            '7T_L':   f'7T-06-L-Hippocampus-{alg}',
            '7T_R':   f'7T-06-R-Hippocampus-{alg}',
        }
        has_dr = cols['3Tdr_L'] in df.columns and cols['3Tdr_R'] in df.columns

        long_data = []
        for _, row in df.iterrows():
            for side in ['L', 'R']:
                v3   = row.get(cols[f'3T_{side}'])
                v3dr = row.get(cols[f'3Tdr_{side}']) if has_dr else None
                v7   = row.get(cols[f'7T_{side}'])
                if pd.notna(v3) and pd.notna(v7):
                    long_data.append({
                        'Algorithm': alg,
                        'Subject': row['Folder'],
                        'Side': side,
                        '3T': v3,
                        '3Tdr': v3dr,
                        '7T': v7,
                    })

        long_df = pd.DataFrame(long_data)
        if long_df.empty:
            continue

        long_df['Diff_3T_minus_7T'] = long_df['3T'] - long_df['7T']
        long_df['AbsDiff'] = long_df['Diff_3T_minus_7T'].abs()

        threshold = long_df['AbsDiff'].mean() + 2.5 * long_df['AbsDiff'].std()
        outliers_df = long_df[long_df['AbsDiff'] > threshold].copy()

        if not outliers_df.empty:
            outliers_df['AbsDiff_Mean'] = long_df['AbsDiff'].mean()
            outliers_df['AbsDiff_Std'] = long_df['AbsDiff'].std()
            outliers_df['AbsDiff_Threshold'] = threshold
            outliers_rows.append(outliers_df)
            all_outlier_subjects.update(outliers_df['Subject'].unique())

    # --- Pass 2: Recompute stats excluding any subject with an outlier ---
    results = []
    for alg in sorted(algorithms):
        cols = {
            '3T_L':   f'3T-10-L-Hippocampus-{alg}',
            '3T_R':   f'3T-10-R-Hippocampus-{alg}',
            '3Tdr_L': f'3T-10dr-L-Hippocampus-{alg}',
            '3Tdr_R': f'3T-10dr-R-Hippocampus-{alg}',
            '7T_L':   f'7T-06-L-Hippocampus-{alg}',
            '7T_R':   f'7T-06-R-Hippocampus-{alg}',
        }
        has_dr = cols['3Tdr_L'] in df.columns and cols['3Tdr_R'] in df.columns

        long_data = []
        for _, row in df.iterrows():
            if row['Folder'] in all_outlier_subjects:
                continue  # skip entire subject
            for side in ['L', 'R']:
                v3   = row.get(cols[f'3T_{side}'])
                v3dr = row.get(cols[f'3Tdr_{side}']) if has_dr else None
                v7   = row.get(cols[f'7T_{side}'])
                if pd.notna(v3) and pd.notna(v7):
                    long_data.append({
                        'Algorithm': alg,
                        '3T': v3,
                        '3Tdr': v3dr,
                        '7T': v7,
                    })

        long_df = pd.DataFrame(long_data)
        if long_df.empty:
            continue

        n = len(long_df)
        mean_3T = long_df['3T'].mean()
        std_3T = long_df['3T'].std()
        mean_7T = long_df['7T'].mean()
        std_7T = long_df['7T'].std()

        diffs_3T = long_df['7T'] - long_df['3T']
        abs_mean_diff_3T = diffs_3T.abs().mean()

        shapiro_3T_p = shapiro(diffs_3T)[1] if len(diffs_3T) >= 3 else None
        t_p_3T = ttest_rel(long_df['7T'], long_df['3T'])[1]
        try:
            w_p_3T = wilcoxon(long_df['7T'], long_df['3T'])[1]
        except ValueError:
            w_p_3T = None

        if has_dr and long_df['3Tdr'].notna().all():
            mean_3Tdr = long_df['3Tdr'].mean()
            std_3Tdr = long_df['3Tdr'].std()
            diffs_3Tdr = long_df['7T'] - long_df['3Tdr']
            abs_mean_diff_3Tdr = diffs_3Tdr.abs().mean()
            shapiro_3Tdr_p = shapiro(diffs_3Tdr)[1] if len(diffs_3Tdr) >= 3 else None
            t_p_3Tdr = ttest_rel(long_df['7T'], long_df['3Tdr'])[1]
            try:
                w_p_3Tdr = wilcoxon(long_df['7T'], long_df['3Tdr'])[1]
            except ValueError:
                w_p_3Tdr = None
            direction_3Tdr = "⬇️ 7T < 3Tdr" if mean_7T < mean_3Tdr else "⬆️ 7T > 3Tdr"
        else:
            mean_3Tdr = std_3Tdr = abs_mean_diff_3Tdr = shapiro_3Tdr_p = t_p_3Tdr = w_p_3Tdr = direction_3Tdr = None

        def sig(p): return "✅" if p is not None and p < 0.05 else "❌"
        def normal(p): return "✅" if p is not None and p >= 0.05 else "❌"

        results.append({
            'Algorithm': alg,
            'N hippocampi (after global outlier removal)': n,
            'Total outlier subjects removed': len(all_outlier_subjects),
            'Mean 3T': mean_3T,
            'Std 3T': std_3T,
            'Mean 3Tdr': mean_3Tdr,
            'Std 3Tdr': std_3Tdr,
            'Mean 7T': mean_7T,
            'Std 7T': std_7T,
            'Δ Mean (3T - 7T)': abs_mean_diff_3T,
            'Δ Mean (3Tdr - 7T)': abs_mean_diff_3Tdr,
            'Direction (3T vs 7T)': "⬇️ 7T < 3T" if mean_7T < mean_3T else "⬆️ 7T > 3T",
            'Shapiro p (3T vs 7T)': shapiro_3T_p,
            'Shapiro normal? (3T vs 7T)': normal(shapiro_3T_p),
            't-test p (3T vs 7T)': t_p_3T,
            't-test sig? (3T vs 7T)': sig(t_p_3T),
            'Wilcoxon p (3T vs 7T)': w_p_3T,
            'Wilcoxon sig? (3T vs 7T)': sig(w_p_3T),
            'Direction (3Tdr vs 7T)': direction_3Tdr,
            'Shapiro p (3Tdr vs 7T)': shapiro_3Tdr_p,
            'Shapiro normal? (3Tdr vs 7T)': normal(shapiro_3Tdr_p),
            't-test p (3Tdr vs 7T)': t_p_3Tdr,
            't-test sig? (3Tdr vs 7T)': sig(t_p_3Tdr),
            'Wilcoxon p (3Tdr vs 7T)': w_p_3Tdr,
            'Wilcoxon sig? (3Tdr vs 7T)': sig(w_p_3Tdr),
        })

    # Save results
    pd.DataFrame(results).to_csv(stats_output_csv, index=False)
    print(f"✓ Stats saved to {stats_output_csv}")

    # Save outliers detail
    if outliers_rows:
        pd.concat(outliers_rows, ignore_index=True).to_csv(outliers_output_csv, index=False)
        print(f"✓ Outlier details saved to {outliers_output_csv}")
    else:
        print("✓ No outliers detected.")

    print(f"Global outlier subjects removed from all algorithms: {len(all_outlier_subjects)}")


def process_nq_percentiles_stats_no_outliers():
    import os
    import pandas as pd
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    nq_results_csv = os.path.join(working_path, 'nq_results.csv')
    stats_output_csv = os.path.join(working_path, 'stats_nq_percentiles_no_outliers.csv')
    outliers_csv = os.path.join(working_path, 'stats_volumes_outliers.csv')  # from the volume analysis step

    # ---------- load NQ results ----------
    df = pd.read_csv(nq_results_csv)

    # ---------- build paired 3T/7T per PRID ----------
    pairs = []
    for prid, group in df.groupby('PRID'):
        row_3t = group[group['Field Strength'] == 3]
        row_7t = group[group['Field Strength'] == 7]
        if len(row_3t) == 1 and len(row_7t) == 1:
            pairs.append({
                'PRID': prid,
                'Left 3T': row_3t['Left Percentile'].values[0],
                'Right 3T': row_3t['Right Percentile'].values[0],
                'Left 7T': row_7t['Left Percentile'].values[0],
                'Right 7T': row_7t['Right Percentile'].values[0],
            })

    paired_df = pd.DataFrame(pairs)

    # Early exit if nothing to analyze
    if paired_df.empty:
        pd.DataFrame([{
            'N hippocampi': 0,
            'Mean 3T': None, 'Std 3T': None,
            'Mean 7T': None, 'Std 7T': None,
            'Δ Mean (3T - 7T)': None,
            'Direction': None,
            'Shapiro p': None, 'Shapiro normal?': None,
            't-test p': None, 't-test sig?': None,
            'Wilcoxon p': None, 'Wilcoxon sig?': None,
            'PRIDs excluded (global)': 0,
            'PRIDs remaining': 0
        }]).to_csv(stats_output_csv, index=False)
        print(f"✓ No paired 3T/7T rows found. Empty stats written to {stats_output_csv}")
        return

    # ---------- GLOBAL OUTLIER EXCLUSION ----------
    # If the outliers CSV exists, exclude ANY PRID that appears there (subject-level exclusion).
    global_outlier_subjects = set()
    if os.path.exists(outliers_csv):
        try:
            out_df = pd.read_csv(outliers_csv)
            if {'Subject'}.issubset(out_df.columns):
                # Normalize to ints (tolerate padding / stray text)
                def to_int_safe(x):
                    try:
                        return int(str(x).strip())
                    except Exception:
                        return None
                subj_int = out_df['Subject'].values.astype(int)
                global_outlier_subjects = set(subj_int.tolist())
        except Exception as e:
            print(f"Warning: failed to read outliers CSV ({outliers_csv}): {e}")

    # Normalize PRIDs to int for comparison
    def to_int_safe(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    paired_df['PRID_int'] = paired_df['PRID'].map(to_int_safe)
    before_prids = set(paired_df['PRID_int'].dropna().astype(int).unique().tolist())
    if global_outlier_subjects:
        paired_df = paired_df[~paired_df['PRID_int'].isin(global_outlier_subjects)].copy()
    after_prids = set(paired_df['PRID_int'].dropna().astype(int).unique().tolist())
    excluded_prids = sorted(before_prids - after_prids)

    # ---------- Stack left/right to long form ----------
    long_data = []
    for _, row in paired_df.iterrows():
        long_data.append({
            'PRID': row['PRID_int'],
            'Side': 'L',
            '3T': row['Left 3T'],
            '7T': row['Left 7T']
        })
        long_data.append({
            'PRID': row['PRID_int'],
            'Side': 'R',
            '3T': row['Right 3T'],
            '7T': row['Right 7T']
        })

    long_df = pd.DataFrame(long_data).dropna(subset=['3T','7T'])

    if long_df.empty:
        pd.DataFrame([{
            'N hippocampi': 0,
            'Mean 3T': None, 'Std 3T': None,
            'Mean 7T': None, 'Std 7T': None,
            'Δ Mean (3T - 7T)': None,
            'Direction': None,
            'Shapiro p': None, 'Shapiro normal?': None,
            't-test p': None, 't-test sig?': None,
            'Wilcoxon p': None, 'Wilcoxon sig?': None,
            'PRIDs excluded (global)': len(excluded_prids),
            'PRIDs remaining': 0
        }]).to_csv(stats_output_csv, index=False)
        print(f"✓ After global outlier exclusion, no data remained. Stats written to {stats_output_csv}")
        if excluded_prids:
            print(f"Excluded PRIDs: {excluded_prids}")
        return

    diffs = long_df['3T'] - long_df['7T']

    # ---------- Stats ----------
    mean_3T = long_df['3T'].mean()
    std_3T = long_df['3T'].std()
    mean_7T = long_df['7T'].mean()
    std_7T = long_df['7T'].std()
    delta_mean = mean_3T - mean_7T
    n = len(diffs)

    shapiro_p = shapiro(diffs)[1] if len(diffs) >= 3 else None
    t_p = ttest_rel(long_df['3T'], long_df['7T'])[1]
    try:
        w_p = wilcoxon(long_df['3T'], long_df['7T'])[1]
    except ValueError:
        w_p = None

    # Helpers
    def sig(p): return "✅" if p is not None and p < 0.05 else "❌"
    def normal(p): return "✅" if p is not None and p >= 0.05 else "❌"
    direction = "⬇️ 7T < 3T" if mean_7T < mean_3T else "⬆️ 7T > 3T"

    # ---------- Output ----------
    results = [{
        'N hippocampi': n,
        'Mean 3T': mean_3T,
        'Std 3T': std_3T,
        'Mean 7T': mean_7T,
        'Std 7T': std_7T,
        'Δ Mean (3T - 7T)': delta_mean,
        'Direction': direction,
        'Shapiro p': shapiro_p,
        'Shapiro normal?': normal(shapiro_p),
        't-test p': t_p,
        't-test sig?': sig(t_p),
        'Wilcoxon p': w_p,
        'Wilcoxon sig?': sig(w_p),
        'PRIDs excluded (global)': len(excluded_prids),
        'PRIDs remaining': len(after_prids)
    }]

    stats_df = pd.DataFrame(results)
    stats_df.to_csv(stats_output_csv, index=False)
    print(f"✓ Stats (with GLOBAL outlier subject exclusion) saved to {stats_output_csv}")
    if excluded_prids:
        print(f"Excluded PRIDs: {excluded_prids}")


def process_demographics_no_outliers():
    import os, re
    import pandas as pd

    # Paths
    working_path = os.path.join(r'/media/neurorad/Extreme Pro/working/3t-7t-hipposeg-working')
    csv_path = os.path.join(working_path, 'data.csv')
    outliers_csv = os.path.join(working_path, 'stats_volumes_outliers.csv')

    # --- helpers ---
    def digits_to_int_or_none(x):
        s = re.sub(r'[^0-9]', '', str(x))
        return int(s) if s else None

    # --- read demographics ---
    df = pd.read_csv(csv_path)

    # Drop rows missing required fields
    df = df.dropna(subset=['PatientResearchID', 'PatientSex', 'PatientBirthDate', 'StudyDate'])

    # Convert date fields safely
    df['BirthDate'] = pd.to_datetime(df['PatientBirthDate'], format='%Y%m%d', errors='coerce')
    df['StudyDate'] = pd.to_datetime(df['StudyDate'], format='%Y%m%d', errors='coerce')

    # Drop rows where parsing failed
    df = df.dropna(subset=['BirthDate', 'StudyDate'])

    # For each unique PatientResearchID, keep only one row — use earliest study date for age calc
    df_unique = df.sort_values('StudyDate').groupby('PatientResearchID', as_index=False).first()

    # --- GLOBAL OUTLIER EXCLUSION (from stats_volumes_outliers.csv) ---
    global_outlier_subjects = set()
    if os.path.exists(outliers_csv):
        try:
            out_df = pd.read_csv(outliers_csv)
            if {'Subject'}.issubset(out_df.columns):
                # Normalize to ints (tolerate padding / stray text)
                def to_int_safe(x):
                    try:
                        return int(str(x).strip())
                    except Exception:
                        return None
                subj_int = out_df['Subject'].values.astype(int)
                global_outlier_subjects = set(subj_int.tolist())
        except Exception as e:
            print(f"Warning: failed to read outliers CSV ({outliers_csv}): {e}")

    # Normalize PatientResearchID to int and exclude matches
    df_unique['PRID_int'] = df_unique['PatientResearchID'].map(digits_to_int_or_none)

    before_n_patients = len(df_unique)
    if global_outlier_subjects:
        df_unique = df_unique[~df_unique['PRID_int'].isin(global_outlier_subjects)].copy()
    after_n_patients = len(df_unique)
    excluded_n = before_n_patients - after_n_patients

    # Number of unique patients
    num_patients = after_n_patients

    # Percent female
    percent_female = (df_unique['PatientSex'].astype(str).str.upper().str[0] == 'F').mean() * 100 if num_patients else 0.0

    # Compute age in years
    if num_patients:
        df_unique['Age'] = (df_unique['StudyDate'] - df_unique['BirthDate']).dt.days / 365.25
        mean_age = df_unique['Age'].mean()
        min_age = df_unique['Age'].min()
        max_age = df_unique['Age'].max()
    else:
        mean_age = min_age = max_age = float('nan')

    # Output
    if excluded_n:
        print(f"✓ Global outlier exclusion: removed {excluded_n} patient(s) found in {os.path.basename(outliers_csv)}")
    print(f"✓ Number of patients: {num_patients}")
    print(f"✓ Percent female: {percent_female:.1f}%")
    if num_patients:
        print(f"✓ Age range: {min_age:.1f} to {max_age:.1f} years")
        print(f"✓ Mean age: {mean_age:.1f} years")
    else:
        print("✓ Age stats: no patients remaining after exclusion")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    else:
        return f"{seconds / 86400:.2f} days"  


if __name__ == '__main__':
    preprocess_no_mni_extraction_contrast()

    
import configparser
import os
import re
import shutil
import subprocess
import sys
import time

import ants
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

working_path = os.path.join(os.getcwd(), r'working/hippocampal-seg-working')
nii_path = os.path.join(working_path, 'nii')
t1_mni_path = os.path.join(os.getcwd(), r'mri_brain_registration/templates/MNI/t1.nii.gz')

def segment_fastsurfer(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0    
    print(f"Processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        prid_path = os.path.join(nii_path, prid_folder)
        start_time = time.time()
        if reprocess:
            for file in os.listdir(prid_path):
                file_path = os.path.join(nii_path, prid_folder, file)
                if 'aparc' in file or 'fastsurfer' in file:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

        seg_tmp_path = os.path.join(prid_path, 'fastsurfer')
        # Clean up tmp path
        if os.path.exists(seg_tmp_path):
            shutil.rmtree(seg_tmp_path)

        asegdkt_segfile_nii_path = os.path.join(prid_path, 'aparc.DKTatlas+aseg.deep.nii.gz')
        study_path = os.path.join(nii_path, prid_folder)
        if not os.path.exists(asegdkt_segfile_nii_path):
            t1_path = os.path.join(study_path, 'T1.nii.gz')
            os.mkdir(seg_tmp_path)
            
            # Quick segmentation
            subject_id = prid_folder
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
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

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
            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-fastsurfer-label.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-fastsurfer-label.nii.gz')
            combined_seg_path = os.path.join(study_path, 'hippocampi-fastsurfer.seg.nrrd')
            segfile_path = os.path.join(study_path, 'aparc.DKTatlas+aseg.deep.nii.gz')
            
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                # Isolate hippocampi from wmparc.DKTatlas.mapped.nii.gz
                # 17 = left hippocampus
                # 53 = right hippocampus            
                seg = sitk.ReadImage(segfile_path)
                seg_data = sitk.GetArrayFromImage(seg)            
                r_hippo_data = np.zeros_like(seg_data)
                #r_hippo_indices = np.where(seg_data == 53)
                r_hippo_indices = np.where(seg_data == 1710)
                r_hippo_data[r_hippo_indices] = 1
                r_hippo = sitk.GetImageFromArray(r_hippo_data)
                r_hippo.CopyInformation(seg)

                l_hippo_data = np.zeros_like(seg_data)
                #l_hippo_indices = np.where(seg_data == 17)
                l_hippo_indices = np.where(seg_data == 549)
                l_hippo_data[l_hippo_indices] = 1
                l_hippo = sitk.GetImageFromArray(l_hippo_data)
                l_hippo.CopyInformation(seg)

                sitk.WriteImage(r_hippo, r_hippo_path)
                sitk.WriteImage(l_hippo, l_hippo_path)

        end_time = time.time()
        elapsed_time = end_time - start_time  # Time taken for one study
        total_time += elapsed_time  # Total time so far
        avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
        remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
        estimated_time_left = avg_time_per_study * remaining_studies
        print(f"Processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")

def segment_hippodeep(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"hippodeep processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(nii_path, prid_folder, file)
                    if 'hippodeep' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Just isolate the hippocampus label
            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-hippodeep-prob.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-hippodeep-prob.nii.gz')
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                # Define the paths
                script_path = "/home/neurorad/radiplab/hippodeep_pytorch/deepseg1.sh"
                t1_path = os.path.join(study_path, 'T1.nii.gz')
                output_path = os.path.join(study_path, 'hippodeep')
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
                sitk.WriteImage(r_prob_mask_image_rescaled, r_hippo_path)

                l_prob_mask_path = os.path.join(output_path, 'T1_mask_L.nii.gz')
                l_prob_mask_image = sitk.ReadImage(l_prob_mask_path)
                l_prob_mask_array = sitk.GetArrayFromImage(l_prob_mask_image)
                l_prob_mask_array = l_prob_mask_array.astype(np.float32)
                l_prob_mask_array_rescaled = np.interp(l_prob_mask_array, (np.min(l_prob_mask_array), np.max(l_prob_mask_array)), (0, 1))
                l_prob_mask_image_rescaled = sitk.GetImageFromArray(l_prob_mask_array_rescaled)
                l_prob_mask_image_rescaled.CopyInformation(l_prob_mask_image)
                sitk.WriteImage(l_prob_mask_image_rescaled, l_hippo_path)

                thresholds = [1e-6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                for threshold in thresholds: # Loop over each threshold and create/save the corresponding label image
                    # Create a binary mask for the current threshold
                    r_thresholded_label = sitk.BinaryThreshold(r_prob_mask_image_rescaled, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                    l_thresholded_label = sitk.BinaryThreshold(l_prob_mask_image_rescaled, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                    
                    # Define the file name using the current threshold
                    r_label_filename = 'r-hippocampus-hippodeep-' + str(threshold) + '-label.nii.gz'
                    r_label_path = os.path.join(study_path, r_label_filename)
                    l_label_filename = 'l-hippocampus-hippodeep-' + str(threshold) + '-label.nii.gz'
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


def segment_e2dhipseg(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"segment_e2dhipseg processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(nii_path, prid_folder, file)
                    if 'e2dhipseg' in file and file != 'e2dhipseg_masks':
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-e2dhipseg-prob.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-e2dhipseg-prob.nii.gz')
            
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                e2d_result_path = os.path.join(study_path, 'e2dhipseg_masks')
                if not os.path.exists(e2d_result_path):
                    # Define the paths
                    t1_path = os.path.join(study_path, 'T1.nii.gz')

                    python_executable = "/home/neurorad/anaconda3/envs/radiology/bin/python3"
                    e2dhipseg_path = "/home/neurorad/radiplab/e2dhipseg"
                    script_path = os.path.join(e2dhipseg_path, "run.py")
                    command = [python_executable, script_path, t1_path, '-reg']
                    result = subprocess.run(command, cwd=e2dhipseg_path) # output is in ./e2dhipseg_masks
                
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

                    sitk.WriteImage(r_e2d_sitk, r_hippo_path)
                    sitk.WriteImage(l_e2d_sitk, l_hippo_path)

                    # Now save 11 labels at different threshold levels
                    thresholds = [1e-6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                    
                    # Loop over each threshold and create/save the corresponding label image
                    for threshold in thresholds:
                        # Create a binary mask for the current threshold
                        r_thresholded_label = sitk.BinaryThreshold(r_e2d_sitk, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                        l_thresholded_label = sitk.BinaryThreshold(l_e2d_sitk, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
                        
                        # Define the file name using the current threshold
                        r_label_filename = 'r-hippocampus-e2dhipseg-' + str(threshold) + '-label.nii.gz'
                        r_label_path = os.path.join(study_path, r_label_filename)
                        l_label_filename = 'l-hippocampus-e2dhipseg-' + str(threshold) + '-label.nii.gz'
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

def segment_HippMapp3r(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"segment_hippmapper processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(nii_path, prid_folder, file)
                    if 'hippmapper' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-hippmapper-label.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-hippmapper-label.nii.gz')
            
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                hippmapper_working_path = os.path.join(study_path, 'hippmapper_working')

                t1_path = os.path.join(study_path, 'hippmapper_working', 'T1.nii.gz')
                if not os.path.exists(hippmapper_working_path):
                    # Example command: hippmapper seg_hipp --t1w /home/neurorad/radiplab/radip6/working/hippocampal-seg-working/nii/00000001/hippmapper_working/T1.nii.gz
                    os.mkdir(hippmapper_working_path)
                    shutil.copy2(os.path.join(study_path, 'T1.nii.gz'), t1_path)

                    # Reformat T1
                    # Perform skull stripping
                    mni_t1_template_path = 'MNI/1mm/mni_icbm152_t1_tal_nlin_sym_09a.nii'
                    mni_brain_mask_path = 'MNI/1mm/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii'

                    # Step 1: Load the T1 image and MNI template and brain mask
                    t1_image = ants.image_read(t1_path)
                    mni_template = ants.image_read(mni_t1_template_path)
                    mni_brain_mask = ants.image_read(mni_brain_mask_path)

                    # Step 2: Register T1 to the MNI template (Affine + Non-linear registration)
                    transform = ants.registration(fixed=mni_template, moving=t1_image, type_of_transform='SyN')

                    # Step 3: Warp the MNI brain mask back to T1 space using the inverse of the transform
                    warped_brain_mask = ants.apply_transforms(fixed=t1_image, moving=mni_brain_mask, transformlist=transform['invtransforms'], interpolator='nearestNeighbor')

                    # Step 4: Apply the warped brain mask to the T1 image to skull-strip
                    brain_only_t1 = t1_image * warped_brain_mask

                    # Step 5: Save the skull-stripped image (relative path)
                    output_path = os.path.join(os.path.dirname(t1_path), 't1_skull_stripped.nii.gz')
                    ants.image_write(brain_only_t1, output_path)

                    # Step 6: Convert the skull-stripped image to RPI orientation using FreeSurfer's mri_convert
                    # Bizarre...hippmapper wants LPI or RPI, and when I mri_convert, hippmapper interprets it as the opposite
                    # So RAS with mri_convert will be LPI for hippmapper, and then it works
                    output_path_rpi = os.path.join(os.path.dirname(t1_path), 't1_skull_stripped_RAS.nii.gz')
                    command = f"mri_convert --out_orientation RAS {output_path} {output_path_rpi}"
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
                label_map_path = os.path.join(study_path, 'hippmapper_working', 't1_skull_stripped_RAS_T1acq_hipp_pred.nii.gz')

                # Convert back to T1 orientation
                label_map_converted_path = os.path.join(study_path, 'hippmapper_working', 'predictions-converted.nii.gz')                
                subprocess.call([
                    r'/usr/local/freesurfer/7.3.2/bin/mri_convert',
                    '--reslice_like', t1_path,  # Match size and voxel grid of the T1 template
                    '--resample_type', 'nearest',  # Use nearest-neighbor interpolation to preserve mask values
                    label_map_path,  # Input segmentation file
                    label_map_converted_path  # Output file
                ]) 

                label_map_image = sitk.ReadImage(label_map_converted_path)

                # Step 1: Generate the right hippocampus label map (label = 1)
                right_hippo_image = sitk.BinaryThreshold(label_map_image, lowerThreshold=32833, upperThreshold=32833, insideValue=1, outsideValue=0)

                # Step 2: Generate the left hippocampus label map (label = 2)
                left_hippo_image = sitk.BinaryThreshold(label_map_image, lowerThreshold=65535, upperThreshold=65535, insideValue=1, outsideValue=0)

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

def segment_QuickNat(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"segment_quicknat processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(nii_path, prid_folder, file)
                    if 'quicknat' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-quicknat-label.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-quicknat-label.nii.gz')
            
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                working_path = os.path.join(study_path, 'quicknat_working')
                os.mkdir(working_path)

                # quicknat requests running mri_convert conform first
                subject_working_path = os.path.join(working_path, prid_folder)
                os.mkdir(subject_working_path)
                mri_path = os.path.join(subject_working_path, 'mri')
                os.mkdir(mri_path)
                t1_path = os.path.join(study_path, 'T1.nii.gz')
                t1_conformed_path = os.path.join(mri_path, 'orig.mgz')
                command = f"mri_convert --conform {t1_path} {t1_conformed_path}"
                process = subprocess.run(command, shell=True, executable='/bin/bash')

                label_path = os.path.join(study_path, 'aparc.DKTatlas+aseg.deep.nii.gz')
                label_conformed_path = os.path.join(working_path, prid_folder + '_glm.mgz')
                command = f"mri_convert --conform {label_path} {label_conformed_path}"
                process = subprocess.run(command, shell=True, executable='/bin/bash')

                # Modify settings_eval_orig.ini to specify settings
                ini_file_path = '/home/neurorad/radiplab/quickNAT_pytorch/settings_eval_orig.ini'
                config = configparser.ConfigParser()
                config.read(ini_file_path)
                config['EVAL_BULK']['device'] = "0"
                config['EVAL_BULK']['data_dir'] = f'"{working_path}"'
                config['EVAL_BULK']['directory_struct'] = '"FS"'
                config['EVAL_BULK']['save_predictions_dir'] = f'"{working_path}"'
                new_ini_file_path = '/home/neurorad/radiplab/quickNAT_pytorch/settings_eval.ini'
                if os.path.exists(new_ini_file_path):
                    os.remove(new_ini_file_path)
                with open(new_ini_file_path, 'w') as configfile:
                    config.write(configfile)

                # Modify settings_orig.ini to specify settings
                ini_file_path = '/home/neurorad/radiplab/quickNAT_pytorch/settings_orig.ini'
                eval_model_path = "saved_models/finetuned_alldata_axial.pth.tar"
                volumes_txt_path = '/home/neurorad/radiplab/quickNAT_pytorch/test_list.txt'
                config = configparser.ConfigParser()
                config.read(ini_file_path)
                config['COMMON']['device'] = "0"
                config['EVAL']['eval_model_path'] = f'"{eval_model_path}"'
                config['EVAL']['data_dir'] = f'"{working_path}"'
                config['EVAL']['label_dir'] = f'"{working_path}"'
                config['EVAL']['volumes_txt_file'] = f'"{volumes_txt_path}"'
                config['EVAL']['orientation'] = f'"AXI"'
                config['EVAL']['save_predictions_dir'] = f'"{working_path}"'
                new_ini_file_path = '/home/neurorad/radiplab/quickNAT_pytorch/settings.ini'
                if os.path.exists(new_ini_file_path):
                    os.remove(new_ini_file_path)
                with open(new_ini_file_path, 'w') as configfile:
                    config.write(configfile)

                # Modify test_list.txt
                file_contents = prid_folder
                with open(volumes_txt_path, 'w') as file:
                    file.write(file_contents)
                
                # Run quicknat
                quicknat_env_path = '/home/neurorad/anaconda3/envs/quicknat'
                quicknat_path = '/home/neurorad/radiplab/quickNAT_pytorch'
                run_script_path = os.path.join(quicknat_path, 'run.py')
                command = f"source /home/neurorad/anaconda3/bin/activate quicknat && python {run_script_path} --mode=eval"
                process = subprocess.run(command, shell=True, cwd=quicknat_path, executable='/bin/bash')

                # Segmentation is at prid_folder.mgz in quicknat_working
                qn_mgz_path = os.path.join(working_path, prid_folder + '.mgz')
                qn_nii_path = os.path.join(working_path, 'seg.nii.gz')
                subprocess.run(['mri_convert', qn_mgz_path, qn_nii_path])

                t1_conformed_nii_path = os.path.join(working_path, 'T1-conformed.nii.gz')
                subprocess.run(['mri_convert', t1_conformed_path, t1_conformed_nii_path])

                seg = sitk.ReadImage(qn_nii_path)
                t1 = sitk.ReadImage(t1_conformed_nii_path)
                seg.SetOrigin(t1.GetOrigin())
                seg_array = sitk.GetArrayFromImage(seg)  # This will be in z, y, x order
                corrected_array = np.transpose(seg_array, (1, 2, 0))  # Rearrange the axes
                corrected_array = np.flip(corrected_array, axis=1)  # Flip along height
                corrected_array = np.flip(corrected_array, axis=2)  # Flip along height
                corrected_array = np.flip(corrected_array, axis=1)  # Flip along height
                corrected_array = np.flip(corrected_array, axis=2)
                corrected_image = sitk.GetImageFromArray(corrected_array)
                corrected_image.CopyInformation(seg)

                # Put back into T1 original space
                resample = sitk.ResampleImageFilter()
                t1_orig = sitk.ReadImage(t1_path)
                resample.SetReferenceImage(t1_orig)
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
                resample.SetOutputDirection(t1_orig.GetDirection())
                resample.SetOutputOrigin(t1_orig.GetOrigin())
                resample.SetOutputSpacing(t1_orig.GetSpacing())
                resample.SetSize(t1_orig.GetSize())
                resampled_image = resample.Execute(corrected_image)
                
                # Save full seg
                resampled_image_path = os.path.join(study_path, 'quicknat-labels.nii.gz')
                sitk.WriteImage(resampled_image, resampled_image_path)                

                # Save final hippocampal segs
                right_hippo = sitk.BinaryThreshold(resampled_image, lowerThreshold=29, upperThreshold=29, insideValue=1, outsideValue=0)
                left_hippo = sitk.BinaryThreshold(resampled_image, lowerThreshold=14, upperThreshold=14, insideValue=1, outsideValue=0)
                sitk.WriteImage(right_hippo, r_hippo_path)
                sitk.WriteImage(left_hippo, l_hippo_path)

                shutil.rmtree(working_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"quicknat processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    

def segment_AssemblyNet(reprocess=False):
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"segment_AssemblyNet processing {total_studies} studies...")
    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)
            if reprocess:
                for file in os.listdir(study_path):
                    file_path = os.path.join(nii_path, prid_folder, file)
                    if 'assemblynet' in file:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

            # Define paths
            r_hippo_path = os.path.join(study_path, 'r-hippocampus-assemblynet-label.nii.gz')
            l_hippo_path = os.path.join(study_path, 'l-hippocampus-assemblynet-label.nii.gz')
            
            if not os.path.exists(r_hippo_path) or not os.path.exists(l_hippo_path):
                working_path = os.path.join(study_path, 'assemblynet_working')
                os.mkdir(working_path)

                t1_filename = 'T1.nii.gz'
                t1_path = os.path.join(working_path, t1_filename)
                shutil.copy2(os.path.join(study_path, t1_filename), t1_path)
                docker_command = [
                    "sudo", "docker", "run", "--rm", "--gpus", '"device=0"',
                    "-v", f"{working_path}:/data", "volbrain/assemblynet:1.0.0", f"/data/{t1_filename}"
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
                shutil.copy2(os.path.join(working_path, 'native_structures_T1.nii.gz'), assemblynet_labels_path)

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

                shutil.rmtree(working_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"AssemblyNet processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    


def step_1_process_seg_volumes():
    """
    This is the main portion of the segmentation assessment. It adds gray matter, subtracts CSF,
    and subtracts enhancement, recording the volumes for each step, for each case.
    """
    total_studies = len(os.listdir(nii_path))
    total_time = 0

    print(f"stats processing {total_studies} studies...")
    volumes_csv_path = os.path.join(working_path, 'volumes.csv')
    if os.path.exists(volumes_csv_path):
        os.remove(volumes_csv_path)

    for i, prid_folder in enumerate(sorted(os.listdir(nii_path))):
        if prid_folder.startswith('0'): # Avoid .DS_Store
            start_time = time.time()
            study_path = os.path.join(nii_path, prid_folder)

            processed_qi_path = os.path.join(study_path, 'processed_qi')
            if os.path.exists(processed_qi_path):
                shutil.rmtree(processed_qi_path)
            os.mkdir(processed_qi_path)

            # Create the volumes CSV if needed
            if not os.path.exists(volumes_csv_path):
                # Initialize the dataframe with the first column as "PRID"
                df = pd.DataFrame(columns=['PRID'])
                all_columns = set(df.columns)
                
                # Loop through all files in the directory
                for filename in sorted(os.listdir(study_path)):
                    if filename.endswith('-label.nii.gz') and not 'hippodeep-1.0' in filename:
                        # Extract base filename without '.nii.gz'
                        base_filename = filename[:-7]
                        
                        # Define the column names for this file
                        columns = [f'{base_filename} Volume (mL)', 
                                   f'{base_filename} GM Added', 
                                   f'{base_filename} CSF Removed', 
                                   f'{base_filename} Enhancement Removed', 
                                   f'{base_filename} CSF Enhancement Removed',
                                   f'{base_filename} Total Correction']
                        
                        all_columns.update(columns)
                        
                # Convert the set of all columns into a sorted list
                all_columns = sorted(all_columns)

                # Now, re-create the dataframe with all columns added at once
                df = pd.DataFrame(columns=all_columns)
                
                # Save the dataframe to a CSV file
                df.to_csv(volumes_csv_path, index=False)

                df = pd.read_csv(volumes_csv_path)
                
                # Get the sorted folder names from nii_path
                folder_names = sorted([folder for folder in os.listdir(nii_path) if os.path.isdir(os.path.join(nii_path, folder))])
                
                # Create a DataFrame with the folder names to add
                new_folders_df = pd.DataFrame(folder_names, columns=['PRID'])
                
                # Append the new folder names under the 'PRID' column
                df = pd.concat([df, new_folders_df], ignore_index=True)
                
                # Save the updated DataFrame back to the CSV
                df.to_csv(volumes_csv_path, index=False)

            # ***** Create aggregate (agg) shell for each hippocampus - used for processing later
            # Exclude hippocampus-hippodeep-1.0-label - too decimated
            t1 = sitk.ReadImage(os.path.join(study_path, 'T1.nii.gz'))
            voxel_size = t1.GetSpacing()  # Returns a tuple (x_size, y_size, z_size)
            voxel_volume_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2] # Calculate the volume of a single voxel in mm^3
            voxel_volume_ml = voxel_volume_mm3 * 1e-3
            t1_data = sitk.GetArrayFromImage(t1)

            r_agg_shell_data = np.zeros_like(t1_data)
            l_agg_shell_data = np.zeros_like(t1_data)
            for label_file in sorted(os.listdir(study_path)):
                if label_file.endswith('-label.nii.gz') and not 'hippodeep-1.0' in label_file:
                    base_filename = label_file[:-7]
                    label_path = os.path.join(study_path, label_file)
                    label_image = sitk.ReadImage(label_path)
                    label_data = sitk.GetArrayFromImage(label_image)
                    if label_file.startswith('r-'):
                        r_agg_shell_data = r_agg_shell_data + label_data
                    if label_file.startswith('l-'):
                        l_agg_shell_data = l_agg_shell_data + label_data
            
            # Remove voxels shared by all the segs
            smallest_hippos_label = np.zeros_like(t1_data)
            max_value = np.max(r_agg_shell_data)
            smallest_hippos_label[r_agg_shell_data == max_value] = 1
            r_agg_shell_data[r_agg_shell_data == max_value] = 0
            r_agg_shell_data[r_agg_shell_data > 0] = 1

            max_value = np.max(l_agg_shell_data)
            smallest_hippos_label[l_agg_shell_data == max_value] = 1
            l_agg_shell_data[l_agg_shell_data == max_value] = 0
            l_agg_shell_data[l_agg_shell_data > 0] = 1

            r_agg_shell_image = sitk.GetImageFromArray(r_agg_shell_data)
            r_agg_shell_image.CopyInformation(t1)            
            sitk.WriteImage(r_agg_shell_image, os.path.join(processed_qi_path, 'r-agg-shell-labels.nii.gz'))
            l_agg_shell_image = sitk.GetImageFromArray(l_agg_shell_data)
            l_agg_shell_image.CopyInformation(t1)            
            sitk.WriteImage(l_agg_shell_image, os.path.join(processed_qi_path, 'l-agg-shell-labels.nii.gz'))

            # Determine gray matter intensity from smallest hippo seg
            gm_voxels = t1_data[smallest_hippos_label == 1]
            gm_mean = np.mean(gm_voxels)
            gm_std = np.std(gm_voxels)

            # Determine enhancement threshold for contrast subtraction from subtracted image
            subtracted_image = sitk.ReadImage(os.path.join(study_path, 'T1C-T1.nii.gz'))
            subtracted_data = sitk.GetArrayFromImage(subtracted_image)
            brain_mask_path = os.path.join(study_path, 'brain_mask.nii.gz')
            brain_mask_image = sitk.ReadImage(brain_mask_path)
            brain_mask_data = sitk.GetArrayFromImage(brain_mask_image)
            brain_voxels = subtracted_data[brain_mask_data == 1]
            brain_mean = np.mean(brain_voxels)
            brain_std = np.std(brain_voxels)
            enhancement_threshold = brain_mean + brain_std
            enhancing_indices = np.where(subtracted_data > enhancement_threshold)

            # Determine CSF intensity from right lateral ventricle seg
            fs_path = os.path.join(study_path, 'aparc.DKTatlas+aseg.deep.nii.gz')
            fs = sitk.ReadImage(fs_path)
            fs_data = sitk.GetArrayFromImage(fs)
            rv_indices = np.where(fs_data == 1388) # right ventricle
            rv_voxels = t1_data[rv_indices]
            rv_index_set = set(zip(*rv_indices)) # Step 4: Create a mask to remove voxels that are in both rv_indices and enhancing_indices - Convert indices to a set for efficient subtraction
            enhancing_index_set = set(zip(*enhancing_indices))
            remaining_indices = list(rv_index_set - enhancing_index_set) # Subtract enhancing_indices from rv_indices
            remaining_indices = tuple(np.array(i) for i in zip(*remaining_indices)) # Convert remaining_indices back to separate arrays of indices
            rv_voxels_after_subtraction = t1_data[remaining_indices] # Step 5: Get the remaining rv_voxels after subtraction
            rv_mean = np.mean(rv_voxels)
            rv_std = np.std(rv_voxels)
            csf_mean = rv_mean
            csf_std = rv_std           
            
            # Process each label
            for label_file in sorted(os.listdir(study_path)):
                if label_file.endswith('-label.nii.gz') and not 'hippodeep-1.0' in label_file:
                    gray_matter_added_volume = 0
                    csf_removed_volume = 0
                    enhancement_removed_volume = 0
                    total_correction_volume = 0
                    base_filename = label_file[:-7]
                    label_path = os.path.join(study_path, label_file)
                    label_image = sitk.ReadImage(label_path)
                    label_data = sitk.GetArrayFromImage(label_image)

                    # ***** Calculate Volume (mL)
                    total_voxel_count = np.sum(label_data == 1)  # Count the total voxels in the label file
                    total_volume_ml = total_voxel_count * voxel_volume_ml  # Calculate total volume in mL
                    total_volume_ml = round(total_volume_ml, 3)

                    # Add to spreadsheet
                    df = pd.read_csv(volumes_csv_path)  # Load the CSV into a pandas DataFrame
                    volume_column = f'{base_filename} Volume (mL)'  # Column name for Volume (mL)
                    df.loc[df['PRID'] == int(prid_folder), volume_column] = total_volume_ml  # Add volume to the corresponding row
                    df.to_csv(volumes_csv_path, index=False)

                    # ***** Calculate gray matter added
                    # Step 1: Grow the label_data by 1 voxel using binary dilation
                    grown_label_data = ndimage.binary_dilation(label_data, iterations=1)
                    
                    # Step 2: Create the shell by subtracting the original label_data from the grown version
                    gm_added_voxels = grown_label_data.astype(int) - label_data.astype(int)
                    
                    # Step 3: Exclude any voxels where r_agg_shell_data is 0
                    if label_file.startswith('r-'):
                        gm_added_voxels = gm_added_voxels * r_agg_shell_data
                    elif label_file.startswith('l-'):
                        gm_added_voxels = gm_added_voxels * l_agg_shell_data
                    
                    # Step 4: Exclude voxels outside gm_mean ± gm_std on the corresponding t1_data
                    gm_mask = (t1_data >= (gm_mean - gm_std)) & (t1_data <= (gm_mean + gm_std))
                    gm_added_voxels = gm_added_voxels * gm_mask
                    
                    # Step 5: Exclude voxels that are greater than or equal to enhancement_threshold in subtracted_data
                    enhancement_mask = subtracted_data < enhancement_threshold
                    gm_added_voxels = gm_added_voxels * enhancement_mask

                    # Step 6: Keep largest connected component
                    merged_label = (gm_added_voxels | label_data).astype(int)
                    largest_component = keep_largest_connected_component(merged_label)
                    gm_added_voxels = largest_component - label_data
                    gm_added_voxels[gm_added_voxels < 0] = 0  # Ensure no negative values
                    
                    # Calculate volume and add to spreadsheet
                    gm_voxel_count = np.sum(gm_added_voxels == 1) # Count the number of voxels where r_gm_added_data == 1
                    gray_matter_added_volume = gm_voxel_count * voxel_volume_mm3 * 1e-3 # Convert the total volume to mL (1 mm^3 = 1e-3 mL)
                    gray_matter_added_volume = round(gray_matter_added_volume, 3)

                    # Add to spreadsheet
                    df = pd.read_csv(volumes_csv_path) # Load the CSV into a pandas DataFrame
                    gm_added_column = base_filename + ' GM Added' # Construct the column name for 'GM Added'
                    df.loc[df['PRID'] == int(prid_folder), gm_added_column] = gray_matter_added_volume # Find the row where 'PRID' matches prid_folder and update the corresponding value
                    df.to_csv(volumes_csv_path, index=False) # Save the updated DataFrame back to the CSV

                    # Save tmp qi file
                    gm_added_as_2 = np.zeros_like(label_data)
                    gm_added_as_2[gm_added_voxels == 1] = 2 # Set voxels in gm_added_data == 1 to 2 in gm_added_as_2
                    label_data_with_gm = label_data + gm_added_as_2
                    label_data_with_gm_image = sitk.GetImageFromArray(label_data_with_gm)
                    label_data_with_gm_image.CopyInformation(t1)
                    gm_added_filename = label_file.replace('-label.nii.gz', '-gm-label.nii.gz')
                    sitk.WriteImage(label_data_with_gm_image, os.path.join(processed_qi_path, gm_added_filename))
                    x = 5

                    # ***** Calculate CSF removed
                    common_shell_data = None
                    if label_file.startswith('r-'):
                        # Calculate common shell - voxels shared by agg shell and label
                        r_agg_mask = r_agg_shell_data > 0
                        label_mask = label_data == 1
                        common_shell_data = np.where(r_agg_mask & label_mask, 1, 0)
                    if label_file.startswith('l-'):
                        l_agg_mask = l_agg_shell_data > 0
                        label_mask = label_data == 1
                        common_shell_data = np.where(l_agg_mask & label_mask, 1, 0)
                    
                    # Remove CSF intensity from common shell but not overlapping GM intensity
                    not_csf_voxels = np.copy(common_shell_data) # Step 1: Create a copy of common_shell_data to start with for not_csf_voxels
                    csf_mask = (t1_data >= (csf_mean - csf_std)) & (t1_data <= (csf_mean + csf_std)) # Create mask for voxels within csf_mean ± csf_std
                    gm_mask = (t1_data < (gm_mean - gm_std)) | (t1_data > (gm_mean + gm_std)) # Create mask for voxels not in gm_mean ± gm_std
                    combined_mask = csf_mask & gm_mask # Combine the conditions: inside csf range AND outside gm range
                    not_csf_voxels[combined_mask] = 0 # Set voxels in not_csf_voxels to 0 where the combined condition is True
                    csf_voxels = np.zeros_like(common_shell_data) # Step 2: Create csf_voxels as the inverse of not_csf_voxels. Initialize csf_voxels as a zero array with the same shape
                    # Set voxels in csf_voxels based on common_shell_data and not_csf_voxels
                    csf_voxels[(common_shell_data == 1) & (not_csf_voxels == 0)] = 1
                    csf_voxels[(common_shell_data == 1) & (not_csf_voxels == 1)] = 0

                    # Calculate volume and add to spreadsheet
                    voxel_count = np.sum(csf_voxels == 1)
                    csf_removed_volume = voxel_count * voxel_volume_mm3 * 1e-3 # Convert the total volume to mL (1 mm^3 = 1e-3 mL)
                    csf_removed_volume = round(csf_removed_volume, 3)

                    # Add to spreadsheet
                    df = pd.read_csv(volumes_csv_path) # Load the CSV into a pandas DataFrame
                    added_column = base_filename + ' CSF Removed' # Construct the column name
                    df.loc[df['PRID'] == int(prid_folder), added_column] = csf_removed_volume # Find the row where 'PRID' matches prid_folder and update the corresponding value
                    df.to_csv(volumes_csv_path, index=False) # Save the updated DataFrame back to the CSV

                    # Save tmp qi file
                    removed_csf_as_2 = np.zeros_like(label_data)
                    removed_csf_as_2[csf_voxels == 1] = 2 # Set voxels in csf_voxels == 1 to 2 in removed_csf_as_2
                    removed_csf_as_2[(label_data == 1) & (csf_voxels == 0)] = 1 # Set voxels in label_data == 1 and csf_voxels == 0 to 1
                    removed_csf_as_2_image = sitk.GetImageFromArray(removed_csf_as_2)
                    removed_csf_as_2_image.CopyInformation(t1)
                    removed_csf_filename = label_file.replace('-label.nii.gz', '-csf-label.nii.gz')
                    sitk.WriteImage(removed_csf_as_2_image, os.path.join(processed_qi_path, removed_csf_filename))
                    x = 5

                    # ***** Calculate enhancement removed
                    common_shell_data = None
                    if label_file.startswith('r-'):
                        # Calculate common shell - voxels shared by agg shell and label
                        r_agg_mask = r_agg_shell_data > 0
                        label_mask = label_data == 1
                        common_shell_data = np.where(r_agg_mask & label_mask, 1, 0)
                    if label_file.startswith('l-'):
                        l_agg_mask = l_agg_shell_data > 0
                        label_mask = label_data == 1
                        common_shell_data = np.where(l_agg_mask & label_mask, 1, 0)
                    
                    # Remove enhancement from common shell
                    not_enhancing_voxels = np.copy(common_shell_data) # Step 1: Create not_enhancing_voxels from common_shell_data
                    not_enhancing_voxels[subtracted_data >= enhancement_threshold] = 0 # Set voxels in not_enhancing_voxels = 0 if corresponding t1_data voxels are >= enhancing_threshold
                    enhancing_voxels = np.zeros_like(common_shell_data) # Step 2: Create enhancing_voxels as the inverse of not_enhancing_voxels
                    enhancing_voxels[(common_shell_data == 1) & (not_enhancing_voxels == 0)] = 1 # Set voxels in common_shell_data == 1 and not_enhancing_voxels == 0 to 1

                    # Calculate volume and add to spreadsheet
                    voxel_count = np.sum(enhancing_voxels == 1)
                    enhancement_removed_volume = voxel_count * voxel_volume_mm3 * 1e-3 # Convert the total volume to mL (1 mm^3 = 1e-3 mL)
                    enhancement_removed_volume = round(enhancement_removed_volume, 3)

                    # Add to spreadsheet
                    df = pd.read_csv(volumes_csv_path) # Load the CSV into a pandas DataFrame
                    added_column = base_filename + ' Enhancement Removed' # Construct the column name
                    df.loc[df['PRID'] == int(prid_folder), added_column] = enhancement_removed_volume # Find the row where 'PRID' matches prid_folder and update the corresponding value
                    df.to_csv(volumes_csv_path, index=False) # Save the updated DataFrame back to the CSV

                    # Save tmp qi file
                    removed_enhancement_as_2 = np.zeros_like(label_data)
                    removed_enhancement_as_2[enhancing_voxels == 1] = 2 # Set voxels in enhancement_voxels == 1 to 2 in removed_enhancement_as_2
                    removed_enhancement_as_2[(label_data == 1) & (enhancing_voxels == 0)] = 1 # Set voxels in label_data == 1 and enhancement_voxels == 0 to 1
                    removed_enhancement_as_2_image = sitk.GetImageFromArray(removed_enhancement_as_2)
                    removed_enhancement_as_2_image.CopyInformation(t1)
                    removed_enhancement_filename = label_file.replace('-label.nii.gz', '-enhancement-label.nii.gz')
                    sitk.WriteImage(removed_enhancement_as_2_image, os.path.join(processed_qi_path, removed_enhancement_filename))
                    x = 5

                    # Add GM Added, CSF Removed, and Enhancement Removed volumes as before
                    # Calculate 'CSF Enhancement Removed' as the sum of CSF Removed and Enhancement Removed
                    csf_enhancement_removed = csf_removed_volume + enhancement_removed_volume
                    csf_enhancement_removed = round(csf_enhancement_removed, 3)

                    # Add CSF Enhancement Removed to spreadsheet
                    csf_enhancement_column = f'{base_filename} CSF Enhancement Removed'
                    df.loc[df['PRID'] == int(prid_folder), csf_enhancement_column] = csf_enhancement_removed
                    df.to_csv(volumes_csv_path, index=False)

                    # ***** Calculate total volume adjustment
                    total_correction_volume = gray_matter_added_volume + csf_removed_volume + enhancement_removed_volume
                    df = pd.read_csv(volumes_csv_path) # Load the CSV into a pandas DataFrame
                    added_column = base_filename + ' Total Correction' # Construct the column name
                    df.loc[df['PRID'] == int(prid_folder), added_column] = total_correction_volume # Find the row where 'PRID' matches prid_folder and update the corresponding value
                    df.to_csv(volumes_csv_path, index=False) # Save the updated DataFrame back to the CSV

            end_time = time.time()
            elapsed_time = end_time - start_time  # Time taken for one study
            total_time += elapsed_time  # Total time so far
            avg_time_per_study = total_time / (i + 1) # Calculate the average time per MRI based on MRIs processed so far
            remaining_studies = total_studies - (i + 1) # Estimate time left for remaining MRIs
            estimated_time_left = avg_time_per_study * remaining_studies
            print(f"stats processed {prid_folder} ({i+1}/{total_studies}). Estimated time left: {format_time(estimated_time_left)}.")    


def step_2_process_mean_volumes():
    volumes_csv_path = os.path.join(working_path, 'volumes.csv')
    mean_volumes_csv_path = os.path.join(working_path, 'mean_volumes.csv')

    # Generate mean volume stats
    # Create 2nd CSV where rows are labels (combined R and L), and there are 4 columns: GM, CSF, enhancement, total volume change (means)
    if os.path.exists(mean_volumes_csv_path):
        os.remove(mean_volumes_csv_path)
        
    # Load the volumes CSV into a pandas DataFrame
    df = pd.read_csv(volumes_csv_path)
    
    # Create an empty list to store the results
    results = []

    # Identify label columns (excluding 'PRID')
    label_columns = [col for col in df.columns if col != 'PRID']

    # Find unique labels by stripping 'r-' and 'l-' prefixes
    unique_labels = sorted(set(re.sub(r'^[rl]-', '', col.split(' ')[0]) for col in label_columns))

    # Loop through each unique label
    for label in unique_labels:
        # Find the corresponding 'r-' and 'l-' columns for the label
        r_columns = [col for col in label_columns if col.startswith(f'r-{label}')]
        l_columns = [col for col in label_columns if col.startswith(f'l-{label}')]

        # Collect matching columns for GM, CSF, Enhancement, and Total Correction
        categories = ['Volume (mL)', 'GM Added', 'CSF Removed', 'Enhancement Removed', 'CSF Enhancement Removed', 'Total Correction']
        mean_data = {}

        for category in categories:
            # Create exact match strings
            r_exact = f'r-{label} {category}'
            l_exact = f'l-{label} {category}'

            # Ensure we are looking for exact column matches
            r_col = r_exact if r_exact in r_columns else None
            l_col = l_exact if l_exact in l_columns else None

            if r_col and l_col:
                # Calculate the mean of the 'r-' and 'l-' columns for the current category
                mean_value = df[[r_col, l_col]].mean(axis=1).mean()
                mean_data[category] = mean_value

        # Add the results for this label to the list
        results.append({'Label': label, **mean_data})

    # Create a DataFrame with the results and save it to mean_volumes_csv_path
    mean_df = pd.DataFrame(results)
    mean_df.to_csv(mean_volumes_csv_path, index=False)


def keep_largest_connected_component(binary_image):
    # Label connected components in the binary image
    labeled_array, num_features = ndimage.label(binary_image)
    
    if num_features == 0:
        return binary_image  # No components to process
    
    # Find the size of each connected component
    component_sizes = np.bincount(labeled_array.ravel())
    
    # The background component (label 0) is included, so we ignore it by setting its size to 0
    component_sizes[0] = 0
    
    # Find the label of the largest connected component
    largest_component_label = component_sizes.argmax()
    
    # Create a mask for the largest connected component
    largest_component_mask = labeled_array == largest_component_label
    
    return largest_component_mask.astype(int)


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
    step_1_process_seg_volumes()
    
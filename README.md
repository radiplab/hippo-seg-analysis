# hippo-seg-analysis
This is a small repository that contains the code for my voxel intensity analysis of six different hippocampal segmentations. First, there are 6 methods that each implement a hippocampal segmentation algorithm. This was nontrivial so may be of benefit to some. Then, the step_1 method performs the majority of the analysis, adding gray matter and removing CSF and enhancement to assess segmentation accuracy. 

I share this in the interest of transparency to my methods, realizing that actually running the analysis would require curation of a separate precontrast and postcontrast T1 dataset. And of course, the code is provided as-is with no guarantees.

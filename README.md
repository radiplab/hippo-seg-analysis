# hippo-seg-analysis
This is a small repository that contains the code for my analyses of hippocampal segmentations. 

The 3T analysis has been [published](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939414/). It contains code to segment the hippocampi with 6 different methods. This was nontrivial so may be of benefit to some. Then, the step_1 method performs the majority of the analysis, adding gray matter and removing CSF and enhancement to assess segmentation accuracy. 

The 7T analysis performs a very similar analysis on 7T MPRAGE sequences.

The 3T-7T analysis is different. It compares the volumes of hippocampi segmented at 3T versus 7T, and is a more straightforward analysis. QuickNAT was not included due to poor performance at 7T. NeuroQuant was added. I included the code that ran NeuroQuant segmentations, but you'll see it requires aetitle, host, etc information to work.

I share this in the interest of transparency to my methods, realizing that actually running the analysis would require curation of a separate precontrast and postcontrast T1 dataset. And of course, the code is provided as-is with no guarantees.

The analysis requires AntsPy which only runs on Mac and Linux. The segmentations run best on Linux. For example, FastSurfer still uses compoonents of FreeSurfer, and good luck getting that installed on anything but Linux.

# Photograph Watermarker

The intent of this project is to auto-watermark a photo with a predefined signature by overlaying the signature over the most uniform corner of the photo, according to some predefined parameters.

Required Libs:
1) numpy
2) concurrent
3) itertools
4) PIL
5) re

Running the script:
python PhotoWatermarker.py <input 1> <input 2> <input 3> <input 4> <input 5>  > Log.txt

Where:
Input 1: Path to the photo
Input 2: Path to the signature
Input 3: Path to save the watermarked photo (including name)
Input 4: Path to the config file
Input 5: Percentage scale of signature to use for overlay

Concepts used & logic:
1) Total Window Deviation Percentage: For a given window of the photo, calculate the percentage of pixels deviating from the mean/average value
2) Total Row Deviation Percentage: For a given window of the photo, for every row, calculate the percentage of pixels deviating from each row's average. Total row deviation is the average of each row's deviation
3) Total Column Deviation Percentage: For a given window of the photo, for every column, calculate the percentage of pixels deviating from each column's average. Total column deviation is the average of each row's deviation

The signature is optimized based on uniformity to determine exact borders of the signature. The size of new optimized signature is used as the size of the window to be used to sample the portions of the photo for signature overlay

Final Deviation Percentage =
    (Total Window Deviation Percentage * Total Window Deviation Weight in Percentage
     + ((100 - Total Window Deviation Weight in Percentage)
     * (Total Row Deviation Percentage * (Width of New Signature/Height of New Signature)
     + (Total Column Deviation Percentage * (Height of New Signature/Height of New Signature))))) / 100
     
The user specifies a tolerance levels for all the above used parameters. The program will try to overlay the signature on the best suited corner quadrant according to the tolerance levels specified. The program will NOT override the configurations specified by the user, i.e. it will not watermark the photo if the Final Deviation Percentage of all the corner quadrants exceed the tolerance value specified by the user

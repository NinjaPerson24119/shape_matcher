Superpixel Image Segmentation (gSLICr)
This is an external library: https://github.com/carlren/gSLICr
gSLICr is used to reduce the complexity of an image by grouping regions of similar color (these regions are called superpixels).
The configuration file is called superpixel_filter.yaml This configuration file is used by any detectors that utilize the SuperpixelFilter class, which is used to bridge gSLICr with the rest of the vision library.
These are the settings:
gSLICr Settings
Filter Dimensions (input images will be resized to this, potentially increasing GPU load)
gSLICr_width: 1920 (int)
gSLICr_height: 1080 (int)
The range parameters significantly affect the filter's range. count needs to be tuned so that full prop contours will be selected. The higher the number of segments, the less likely there will be distortion (badly formed contours) but also the more likely that prop contours will be split. Split segments can be joined using the automatic filter. The segment number is more of a hint than a rule, so gSLICr will not necessarily use the exact number. Alternatively, the number of superpixels can be deduced by setting a target size (use the flag to switch modes).
gSLICr_noSegs: 2000 (int) (will segfault if set to 400)
gSLICr_spixelSize: 16 (int)
gSLICr_sizeInsteadOfNumber: 0 (int: 0 or 1)
Superpixel connectivity should always be set to true because only closed contours can result in detections. Altering the weight and iterations has not been tested and gSLICr documentation is lacking.
gSLICr_cohWeight: 0.6 (double)
gSLICr_noIters: 5 (int)
gSLICr_doEnforceConnectivity: true (boolean)
Performance
gSLICr is the biggest load in the detection chain. (GPU heavy)
Debug Image Topics
The superpixel filter is implemented as a utility class. Hence it does not have its own debug topics.



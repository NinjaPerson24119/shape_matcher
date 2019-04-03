Automatic Filter
There are two versions of the automatic filter, one for "color analysis" (old version / qualification detectors) and one for "shape analysis" (newer version / shape matcher). This documentation refers to the newer version.
The automatic filter is chained after superpixel image segmentation in order to group split superpixels and also to convert the image into a gray mask. Edge detection is run on this gray mask in order to find contours with OpenCV in a single pass.
The automatic filter uses a 2D histogram with the AB components of the CIE LAB gamut. This means that it will not consider shade (L component) when grouping. To ammend this, simple thresholding (thresholding within an exact range) is used to isolate only complete black or white.
Note that the automatic filter's histogram is not made from the gSLICr average color mask, but the list of superpixels and their colors. Hence, the number of binned items will not correspond to the image area.
The automatic filter uses a clustering algorithm on the 2D histogram, and the colors in each cluster are set to the same color of gray within the input image. This means that if a cluster contains all of the red in an image, for example, all the red will be converted to the same gray. This allows grouping of similarly colored superpixels, but with the disadvantage that it does not consider position or superpixel locality.
When gSLICr is set to a high number of segments, it will help to merge them into complete contours if the automatic filter's 2D histogram is set to a lower number of bins. However, if there is low contrast, a lower number of bins will make it more likely for different colors to end up in the same cluster. If the bins are set too high, similar superpixels will stay separated.
Ideally, gSLICr should be set with a low enough number of segments that minimal contour separation occurs, then the automatic filter should be used with a medium number of bins such that it will fix separations but not aggressively.
Automatic Filter Settings
Each detector has a .yaml file that configures the automatic filter for its specific goals.
The filter bins is as explained above. The black and white margins are how far away to uniformly threshold away from black and white, adding these simple threshold masks directly to the automatic filter's output mask.
shape_detector_auto_filter_bins: 100 (unsigned char)
shape_detector_auto_filter_black_margin: 30 (unsigned char)
shape_detector_auto_filter_white_margin: 30 (unsigned char)
Debug Image Topics
The automatic filter is implemented as a utility class. Hence it does not have its own debug topics.
Performance
The automatic filter is a relatively low load. (CPU heavy)



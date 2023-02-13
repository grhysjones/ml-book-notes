# An Overview of Deep Learning Models Arcitechtures

<details>
<summary><font size=5>Faster-RCNN</font></summary>

- Useful Blogs
    - https://blog.paperspace.com/faster-r-cnn-explained-object-detection/

- R-CNN was a 2014 UC Berkley model that was 3 stage. (1) extract 2k region proposals from an input image (2) compute CNN features on regions and (3) classify regions with SVM. It was a strong model, but it couldn't be trained end-to-end, it has to cache the extracted CNN features requiring many GBs, and each region is fed independently to the CNN making it impossible to run in realtime
- Fast R-CNN was proposed by Facebook AI that included an ROI Pooling layer to extract equal-length feature vectors from all ROIs in the same image. It's a single end-to-end model that shares computations across ROIs. It doesn't require caching of features, so it can run much more quickly and it's more accurate as it turns out. There is no SVM classification, just FC layers after ROI pooling. It still relies on the time consuming selective search algorithm to generate ROIs
- Faster R-CNN is an extension of Fast R-CNN that uses a fully convolutional network with attention to propose ROIs (this is called the Region Proposal Network (RPN)). It also uses multiple reference anchor boxes at different scales and aspect ratios to detect objects. 
    - The RPN is a network, not a selective search algorithm, so it can be trained end-to-end.
    - There are typically 9 anchor boxes that are predicted on (3 for scale and 3 for aspect)
    - The model outputs whether the ROI of the anchor box is backgrd or object
    - The model uses the Intersection-over-Union (IoU) of the anchor box to give an objectiveness score. For the results of the different 9 anchor boxes, an objectiveness class of 1 is given for all that are >0.7. If none are, then choose one that's above 0.5. An objective class of 0 is given for any scored <0.3. Any remaining are not given an objectiveness class and are not used to train the classifier
</details>
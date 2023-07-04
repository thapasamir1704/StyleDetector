# StyleDetector
This repository contains all the files that were used for our AI project 'Style Detector'.

The model we used for classification could not be uploaded due to GIT restrictions. This project allows a user to input an image from their PC or take an image using their webcam, which is then preprocessed and sent to the model for classification. Once, the image has been successfully classified, keywords are generated that closely resemble their current clothing style, and then GOOGLE API is used to provide smart clothing recommendations. This allows a user to get clothing items such as tops, pants, and jackets that closely resemble their current style without the need to spend a lot of time browsing online. They can easily get through to different sites where they can buy related clothing items.

We utilized the DeepFashion Dataset, which is referenced below:

@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
 }

 @ARTICLE{IEEEACstreetphoto2021,  
   author={F. -H. {Huang} and H. -M. {Lu} and Y. -W. {Hsu}},  
   journal={IEEE Access},   
   title={From Street Photos to Fashion Trends: Leveraging User-Provided Noisy Labels for Fashion Understanding},   
   year={2021},  
   volume={},  
   number={},  
   pages={1-1},  
   doi={10.1109/ACCESS.2021.3069245}}

This project aims to rewrite a set of Bundle Adjustment solvers based on QR decomposition without using an optimization problem solver. The improvement approach used in this project is "Square Root Bundle Adjustment"[1]. After implementing this approach, we compared our solvers with Ceres Solver. 

The whole project frame is based on the online VIO Course of DeepBlue Institute. For more details of this project, please refer to my blog: [https://blog.csdn.net/qq_37746927/article/details/135449043](https://blog.csdn.net/qq_37746927/article/details/135449043)

Reference:
[1] Demmel, N., Sommer, C., Cremers, D., & Usenko, V.C. (2021). Square Root Bundle Adjustment for Large-Scale Reconstruction. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11718-11727.


&nbsp;

i used kitti dataset. if you install udacity, cityscapes, etc.., you should see under this page. regardless of the size of image of dataset, you can train and test images. so you use any dataset you want. but if you use supervised learning, or some annotation file, you should make files for each dataset.

&nbsp;

## Install Dataset

### KITTI

if you want to install kitti stereo dataset, excute the code below. so then, directly you can download kitti dataset and many gt images like disparity map or optical flow map and information used by calibration with other sensor like other cam or velodyne or imu.

```
$ sh data/download_dataset.sh kitti
```

&nbsp;

### Udacity

if you want to install udacity dataset, excute the code below. so then, print dataset's url you should visit. and as for other dataset, you can excute same method with a different name.

```
$ sh data/download_dataset.sh udacity
```


### cityscapes

```
$ sh data/download_dataset.sh cityscapes
```

&nbsp;

### BDD100K

```
$ sh data/download_dataset.sh bdd100k
```
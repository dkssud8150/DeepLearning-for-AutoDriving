
&nbsp;

i used kitti dataset. if you install udacity, cityscapes, etc.., you should see under this page. regardless of the size of image of dataset, you can train and test images. so you use any dataset you want. but if you use supervised learning, or some annotation file, you should make files for each dataset.

&nbsp;

&nbsp;

if you want to compare many dataset, you can see the [page](compare_dataset.md).

## Install Dataset

### KITTI

if you want to install kitti stereo dataset, excute the code below. so then, directly you can download kitti dataset and many gt images like disparity map or optical flow map and information used by calibration with other sensor like other cam or velodyne or imu.

```bash
$ sh data/download_dataset.sh kitti
```

&nbsp;

or if you visit directly site for kitti dataset, [raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [semantic label site](https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015)

&nbsp;

### Udacity

if you want to install udacity dataset, excute the code below. so then, print dataset's url you should visit. and as for other dataset, you can excute same method with a different name.

```bash
$ sh data/download_dataset.sh udacity
```


### cityscapes

i downloaded 10000 images for left image, right image. when you download dataset in cityscapes site, you shoud login the site. so i try to approach "wget" package. if you approach the site using wget, you should modify username and password for cookies in "data/download_dataset.sh".

```bash
$ sh data/download_dataset.sh cityscapes
```

&nbsp;

### BDD100K

```bash
$ sh data/download_dataset.sh bdd100k
```


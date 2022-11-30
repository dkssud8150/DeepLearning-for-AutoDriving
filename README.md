# DeepLearning-for-AutoDrving

| term : 2022.12.01 ~ 2023..

All do Computer Vision needed to Autonomous Driving only via Deep-Learning

&nbsp;

if you prepare kitti dataset, click [this url](refer/prepare_dataset.md).

&nbsp;

---

- directories

```txt
DL4AD
    ⊢ config
    ⊢ data
        ⊢ BDD100K
            ⊢ Annotations
            ⊢ ImageSets
            ⊢ JPEGimages
        ⊢ kitti
            ⊢ train
                ⊢ Annotations
                    ⊢ 000000_10.txt
                    ⊢ 000000_11.txt
                    ⊢ 000001_10.txt
                    ⊢ 000001_11.txt
                    ⊢ ...
                ⊢ ImageSets
                    ⊢ images_2.txt
                    ⊢ images_3.txt
                ⊢ JPEGimages
                    ⊢ images_2
                        ⊢ 000000_10.png
                        ⊢ 000000_11.png
                        ⊢ 000001_10.png
                        ⊢ 000001_11.png
                        ⊢ ...
                    ⊢ images_3
                        ⊢ ...
            ⊢ test
                ⊢ ImageSets
                    ⊢ images_2.txt
                    ⊢ images_3.txt
                ⊢ JPEGimages
                    ⊢ images_2
                        ⊢ ...
                    ⊢ images_3
                        ⊢ ...
        ⊢ udacity
            ⊢ ...
        ⊢ cityscapes
            ⊢ ...
        ⊢ download_dataset.sh
    ⊢ refer
        ⊢ flow.md
        ⊢ prepare_dataset.md
        ⊢ train_and_test.md
    ⊢ source
        ⊢ datasets
            ⊢ dataset.py
            ⊢ dataloader.py
            ⊢ my_transform.py
        ⊢ model
            ⊢ backbone.py
            ⊢ models.py
        ⊢ utils
            ⊢ common.py
            ⊢ loss.py
            ⊢ optimizer.py
    ⊢ LICENSE
    ⊢ README.md
    ⊢ test.py
    ∟ train.py
```

&nbsp;

---

```bash
$ git clone https://dkssud8150/DeepLearning-for-AutoDriving.git
$ cd DeepLearning-for-AutoDriving
```

## Train

```bash
$ python train.py <args>
```

&nbsp;

---

## Inference

```bash
$ python infer.py <args>
```

---

&nbsp;

---
---

## Result

<img src="">
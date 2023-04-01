# Dataset

## Directory Structure
```
.
├── image folder (contains all the images)
│   ├── compressed_train_images (Use PCA compressed)
│   ├── enhanced_train_images (Enhance the color contrast)
│   ├── train_images (Original train images)
│   ├── train_images_fusion (Fusion of original, compressed images and the enhanced images)
│   ├── val_images (Original validation images)
├── Memotion2
│   ├── memotion_train.csv (Train labels)
│   ├── memotion_val.csv (Validation labels)
├── Memotion2_test
│   ├── memotion_test.csv (Test labels)
├── Test images
├── answer.txt (Get from format.py, unrelated to the dataset)
├── format.py (Original label reading code)
├── dataset.py (Modified label reading code)
├── load.ipynb (Load the dataset, as well as the preprocessing methods)

```


## Data Preprocessing
1. Color Enhancement
Color enhancement improves the visibility of the image. According to Akbarinia et. al, deep neural networks are sensitive to the color contrast of the image, which is why researches can use color enhancement to improve the performance of their model. We use the same method to enhance the color contrast of the image. In our preprocessing process. We enhance the contrast by 1.3, 1.6 and 1.9 separately. We feed them throught the multimodal and choose the color enhancement that fits the best. We believe this helps to reveal more details in the image.

Akbarinia, Arash, and Karl R. Gegenfurtner. "How is Contrast Encoded in Deep Neural Networks?." arXiv preprint arXiv:1809.01438 (2018).


2. Image Compression
Not every part of the image is important, so we can compress the image to extract the crucial features before feed them to the training model. We use the same method as Li et al. to compress the image through Principle Component Analysis (PCA). Since PCA compression method generally gives better performance on 2D images, after compressing the image, we fuse the compressed image with the original image. 

Li, Jiahao, et al. "Preprocessing Method Comparisons For VGG16 Fast-RCNN Pistol Detection." EPiC Series in Computing 76 (2021): 39-48.

3. Combined Dateset
Variance in dataset is important for a training model to learn the object from various angles and lighting conditions.

Bishop, Christopher M. Neural networks for pattern recognition. Oxford university press, 1995.

In order to add more variance to the training dataset, we combine the original dataset with the compressed dataset, and the enhanced dataset, and get a combined dataset of 21000 images in total. We believe the added variance will help the model to learn the object better.
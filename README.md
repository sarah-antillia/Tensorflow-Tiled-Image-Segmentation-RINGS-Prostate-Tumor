<h2>Tensorflow-Tiled-Image-Segmentation-RINGS-Prostate-Tumor (2024/12/26)</h2>

This is the first experiment of Tiled Image Segmentation for RINGS Prostate Tumor 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and  <a href="https://drive.google.com/file/d/1cFwfM6C-rO9PrhCd6k19CuGWIN2whUxd/view?usp=sharing">
RINGS-Tiled-Prostate-Tumor-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
<b>
RINGS algorithm dataset<br>
</b>
</a>

<br>Please see also: <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-RINGS-Prostate-Tumor">Tensorflow-Image-Segmentation-RINGS-Prostate-Tumor</a>

<br>
<hr>
<b>Actual Tiled Image Segmentation for Images of 1500x1500 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_14_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_14_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_14_2.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_16_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_16_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_16_6.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_10_14_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_10_14_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_10_14_4.png" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Prostate-TumorSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following web site<br>

<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
<b>
RINGS algorithm dataset<br>
</b>
</a>

<br>
Published: 15 April 2021<br>
Version 1<br>
DOI:10.17632/h8bdwrtnr5.1<br>
<br>
<b>Contributors:</b><br>
Massimo Salvi, Martino Bosco, Luca Molinaro, Alessandro Gambella, Mauro Giulio Papotti,<br>
 Udyavara Rajendra Acharya, Filippo Molinari<br>
<br>
<b>Description</b><br>

This repository contains the image dataset and the manual annotations used to develop the RINGS algorithm for automated prostate glands segmentation:<br>
 Salvi M., Bosco M., L. Molinaro, Gambella A., Papotti M., Udyavara Rajendra Acharya, and Molinari F., <br>
"A hybrid deep learning approach for gland segmentation in prostate histopathological images", <br>
Artificial Intelligence in Medicine 2021 (DOI: 10.1016/j.artmed.2021.102076)
<br>
<br>
Background: In digital pathology, the morphology and architecture of prostate glands have been routinely adopted by pathologists to evaluate the presence of cancer tissue. The manual annotations are operator-dependent, error-prone and time-consuming. The automated segmentation of prostate glands can be very challenging too due to large appearance variation and serious degeneration of these histological structures.
Method: A new image segmentation method, called RINGS (Rapid IdentificatioN of Glandural Structures), is presented to segment prostate glands in histopathological images. We designed a novel glands segmentation strategy using a multi-channel algorithm that exploits and fuses both traditional and deep learning techniques. Specifically, the proposed approach employs a hybrid segmentation strategy based on stroma detection to accurately detect and delineate the prostate glands contours.
Results: Automated results are compared with manual annotations and seven state-of-the-art techniques designed for glands segmentation. Being based on stroma segmentation, no performance degradation is observed when segmenting healthy or pathological structures.  Our method is able to delineate the prostate gland of the unknown histopathological image with a dice score of 90.16% and outperforms all the compared state-of-the-art methods.
Conclusions: To the best of our knowledge, the RINGS algorithm is the first fully automated method capable of maintaining a high sensitivity even in the presence of severe glandular degeneration. The proposed method will help to detect the prostate glands accurately and assist the pathologists to make accurate diagnosis and treatment. The developed model can be used to support prostate cancer diagnosis in polyclinics and community care centres. 
<br><br>
<b>Licence</b>: CC BY 4.0<br>


<br>
<h3>
<a id="2">
2 Tiled-Prostate-Tumor ImageMask Dataset
</a>
</h3>
 If you would like to train this Prostate-Tumor Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1cFwfM6C-rO9PrhCd6k19CuGWIN2whUxd/view?usp=sharing">
RINGS-Tiled-Prostate-Tumor-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-Prostate-Tumor
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a 512x512 pixels tiles dataset generated from 1500x1500 pixels IMAGES and MANUAL_TUMOR of TRAIN only.<br>
We excluded all black (empty) masks and their corresponding images to generate our dataset from the original one.<br>  
<pre>
./RINGS
└─TRAIN
    ├─IMAGES
    └─MANUAL TUMOR
</pre>
On the derivation of this tiled dataset, please refer to the following Python scripts.<br>
<li><a href="./generator/TrainTumorImageMaskDatasetGenerator.py">TrainTumorImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_tiled_master.py">split_tiled_master.py</a></li>
<br>
For example, a 1500x1500 pixels image can be split into 512x512 pixels 9 tiles as shown below.<br>
<b>1500x1500 pixels image</b><br>
<table>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/10001.jpg" witdh="606" height="606"></td>
</tr>
</table>
<br>
<b>512x512 pixels tiledly splitted images</b><br>
<table>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_0x0.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_0x1.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_0x2.jpg" witdh="200" height="200"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_1x0.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_1x1.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_1x2.jpg" witdh="200" height="200"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_2x0.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_2x1.jpg" witdh="200" height="200"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/tiles/10001_2x2.jpg" witdh="200" height="200"></td>
</tr>

</table>

<br>
<b>Tiled-Prostate-Tumor Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/Tiled-Prostate-Tumor_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Prostate-Tumor TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Prostate-Tumorand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
normalization  = True
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 7
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0002
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_CUBIC"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = False
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = True
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 1
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 1 image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/epoch_change_tiled_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 29 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/train_console_output_at_epoch_29.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Prostate-Tumor</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Prostate-Tumor.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/evaluate_console_output_at_epoch_29.png" width="720" height="auto">
<br><br>Image-Segmentation-Prostate-Tumor

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Prostate-Tumor/test was not low, and dice_coef not so high as shown below.
<br>
<pre>
loss,0.4375
dice_coef,0.7225
</pre>
<br>

<h3>
5 Tiled inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Prostate-Tumor</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Prostate-Tumor.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_15_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_15_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_15_1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_14_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_14_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_14_2.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_14_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_14_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_14_8.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_15_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_15_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_15_2.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_9_16_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_9_16_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_9_16_2.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/images/P5_D5_10_14_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test/masks/P5_D5_10_14_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Prostate-Tumor/mini_test_output_tiled/P5_D5_10_14_4.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. RINGS algorithm dataset</b><br>
Massimo Salvi, Martino Bosco, Luca Molinaro, Alessandro Gambella, Mauro Giulio Papotti,<br>
 Udyavara Rajendra Acharya, Filippo Molinari<br>
DOI:10.17632/h8bdwrtnr5.1<br>

<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
https://data.mendeley.com/datasets/h8bdwrtnr5/1</a>
<br>



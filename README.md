# Pytorch_Image_Style_Transfer_Gatys

Unofficial Pytorch(1.0+) implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)



If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)

------

## Requirements

- Python 3.6+
- PyTorch 1.0+
- TorchVision
- Pillow

Anaconda environment recommended here!

- **GPU environment!!!** 



## Usage

------

## test

1. Clone this repository 

   ```bash
   git clone https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys
   cd Pytorch_Image_Style_Transfer_Gatys
   ```

   

2. Generate the output image. A transferred output image and a content_output_pair image and a NST_demo_like image will be generated.

   ```python
   python style_transfer.py -c content_image_path -s style_image_path
   ```

   ```
   usage: style_transfer.py [-h] --content CONTENT --style STYLE [--gpu GPU]
                            [--iteration ITERATION]
                            [--snapshot_interval SNAPSHOT_INTERVAL]
                            [--style_weight STYLE_WEIGHT] [--tv_weight TV_WEIGHT]
                            [--lr LR] [--save_dir SAVE_DIR]
   
   Image Style Transfer Using Convolutional Neural Networks, CVPR 2016, by Gatys
   et al.
   
   optional arguments:
     -h, --help            show this help message and exit
     --content CONTENT, -c CONTENT
                           path to content image
     --style STYLE, -s STYLE
                           path to style image
     --gpu GPU, -g GPU     GPU ID(nagative value indicate CPU)
     --iteration ITERATION
                           total no. of iterations of the algorithm, default=100
     --snapshot_interval SNAPSHOT_INTERVAL
                           interval of snapshot to generate image, default=10
     --style_weight STYLE_WEIGHT
                           style loss hyperparameter, default=1000
     --tv_weight TV_WEIGHT
                           total variance loss hyperparameter, default=0.01
     --lr LR               learning rate for L-BFGS, default=1
     --save_dir SAVE_DIR   save directory for result and loss
   
   ```

   output_name  will use the combination of content image name and style image name.

------



# result

Some results of  my cat (called Sora) will be shown here.

![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/rain-princess.jpg)![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/neko_rain-princess.jpg)

![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/1348.jpg)![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/neko_1348.jpg)

![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/candy.jpg)![image](https://github.com/irasin/Pytorch_Image_Style_Transfer_Gatys/blob/master/res/neko_candy.jpg)





# My Opinion

It costs too much time to generate a style-transferred image using optimization-based approach even with a GPU.

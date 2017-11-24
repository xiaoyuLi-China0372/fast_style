# fast photo style transform

This is an implementation based on mxnet-python of the paper
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1508.06576) by Justin Johnson, Alexandre Alahi, Li Fei-Fei.

## How to use

Then run `python fast_style.py`, use `-h` to see more options

### create output and training set directory
* mkdir output
* mkdir datas/training_images

### acquire training image set
download the Coco dataset：http://msvocds.blob.core.windows.net/coco2015/test2015.zip
or download image dataset from challenger.ai

### train
python fast_style.py --train 1 --train_path datas/training_images/

### transform
python fast_style.py --input_image input/xxx.jpg

## Sample results
output/

## Note

* use the mxnet-0.11.0
* The current implementation is based on:
  https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/FastNeuralStyle and https://github.com/apache/incubator-mxnet/tree/master/example/neural-style


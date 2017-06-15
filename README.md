# Weakly Supervised Localization Implementations in Tensorflow
Tensorflow Implementations of 
- Class Activation Maps
- Grad-CAM 'Gradient Based Class Activation Maps'
- Refining of results using Conditional Random Fields

# Class Activation Maps

[Class Activation Maps](http://cnnlocalization.csail.mit.edu/) provides a way of doing weakly supervised localization in Convolutional Neural Networks. This tensorflow Implementation is partly inspired by [this](https://github.com/jazzsaxmafia/Weakly_detector) approach. Results obtained on Imagenet dataset is given below.

_Note:_ The approach here uses the caffe weights provided by the [author](https://github.com/metalbubble/CAM). The weights are converted from caffe to tensorflow's compatible weights using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).
- The model here uses VGG16 architecture.
- For Training you would need the pretrained weights.
- A link for the converted pretrained weight file will be provided soon.
- If you use the code then kindly cite the respective [work](https://arxiv.org/pdf/1512.04150.pdf)

![alt tag](https://github.com/gondal1/weakly_localizations_tensorflow/blob/master/sample_images/cam.png)


# Grad-CAM (Gradient Based Class Activation Maps)
[Grad-CAM](https://arxiv.org/abs/1611.07450) is a recent state of the art approach for weakly supervised localization in CNNs. The general idea behind the approach is same as the CAM. However, instead of using linear combinations of last convolutional layer's feature maps with the output as weights to signify the importance of each feature map, they make use of the gradient backpropagations from the output to last convolutional layer feature maps. This strategy enables them to retain fully connected layer at the end of the network and hence it doesn't affect the classification accuracy.

- They also use guided backpropagation to get the pixelspace visualizations of the output prediction.
- The network architecture is VGG16.
- The weights are here are also converted from Caffe to Tensorflow using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).
- A link for the converted pretrained file weights will be provided soon.
- If you use the code then kindly cite the respective [work](https://arxiv.org/abs/1611.07450)

![alt tag](https://github.com/gondal1/weakly_localizations_tensorflow/blob/master/sample_images/grad_cam.png)

# Refining of results using Conditional Random Fields
[PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) is a python wrapper of Phillip Krähenbühl's dense (fully connected) CRFs with gaussian edge potentials.

- For quick installation of the library
```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
- The provided cam_crf_notebook covers the basic usage of the library, however it is highly recommended to visit [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) for detailed instructions.
- If you use the code then kindly cite the respective work
```
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
Philipp Krähenbühl and Vladlen Koltun
NIPS 2011
```
![alt tag](https://github.com/gondal1/weakly_localizations_tensorflow/blob/master/sample_images/grad_cam_crf.png)
# Requirements
- Python 3.5
- Tensorflow v0.12
- Opencv2

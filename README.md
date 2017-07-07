# python-guided-filter
Numpy/Scipy implementation of the (fast) Guided Filter. Adapted from Kaiming's Matlab code.

gf.py:guided_filter runs the filter on one- or three-channel guide images (I) and filtering inputs (p) with any number of channels (the filter is applied per-channel of p).

Citations:
   * [Guided Image Filtering](http://kaiminghe.com/eccv10/), by Kaiming He, Jian Sun, and Xiaoou Tang, in TPAMI 2013
   * [Fast Guided Filter](https://arxiv.org/abs/1505.00996), by Kaiming He and Jian Sun, arXiv 2015

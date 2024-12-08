# MiniTorch Module 4


# 4.5

![alt text](image.png)

![alt text](image-1.png)

```
Epoch: 1/25, loss: 32.32080803275224, train accuracy: 0.5244444444444445
Epoch: 2/25, loss: 30.188853852648215, train accuracy: 0.5644444444444444
Epoch: 3/25, loss: 28.277807317472575, train accuracy: 0.6377777777777778
Epoch: 4/25, loss: 25.80966769325519, train accuracy: 0.6933333333333334
Epoch: 5/25, loss: 23.103566823584647, train accuracy: 0.7133333333333334
Epoch: 6/25, loss: 20.888525973380904, train accuracy: 0.74
Epoch: 7/25, loss: 17.926669349281546, train accuracy: 0.7644444444444445
Epoch: 8/25, loss: 16.663081308129353, train accuracy: 0.8
Epoch: 9/25, loss: 14.185596682291246, train accuracy: 0.8333333333333334
Epoch: 10/25, loss: 14.867011890132531, train accuracy: 0.82
Epoch: 11/25, loss: 11.989614181801626, train accuracy: 0.8644444444444445
Epoch: 12/25, loss: 12.265498744508024, train accuracy: 0.8222222222222222
Epoch: 13/25, loss: 16.42544683599386, train accuracy: 0.8266666666666667
Epoch: 14/25, loss: 15.686283741064495, train accuracy: 0.8222222222222222
Epoch: 15/25, loss: 11.284489433664438, train accuracy: 0.8466666666666667
Epoch: 16/25, loss: 7.8737201135277735, train accuracy: 0.8711111111111111
Epoch: 17/25, loss: 8.849501059083748, train accuracy: 0.8755555555555555
Epoch: 18/25, loss: 8.46995609648931, train accuracy: 0.8777777777777778
Epoch: 19/25, loss: 8.282765954586521, train accuracy: 0.8644444444444445
Epoch: 20/25, loss: 8.916003819721439, train accuracy: 0.8466666666666667
Epoch: 21/25, loss: 9.133605521472559, train accuracy: 0.8488888888888889
Epoch: 22/25, loss: 8.59349553006427, train accuracy: 0.8822222222222222
Epoch: 23/25, loss: 8.19198319332626, train accuracy: 0.8488888888888889
Epoch: 24/25, loss: 7.1337693457644376, train accuracy: 0.8911111111111111
Epoch: 25/25, loss: 8.042027878334471, train accuracy: 0.8666666666666667
```

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py
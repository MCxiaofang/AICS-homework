# AICS-homework

2023年春季学期 AICS作业存档

## submit分支
**所有实验的需要提交的文件已经整理好放在了submit分支，可以直接提交希冀平台且100分**

256MB左右，太大了，不合并入主分支, 非必须请仅克隆主分支
`git clone -b main git@github.com:MCxiaofang/AICS-homework.git`

其中，5-3实验需要提交的whl文件没有放入仓库，因为其超过了github的最大文件大小50MB限制

`remote: warning: File submit/5-3/tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl is 96.85 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB`
4-1实验需要提交的vgg19_int8.pb同样

`remote: error: File submit/4-1/vgg19_int8.pb is 137.06 MB; this exceeds GitHub's file size limit of 100.00 MB`


03.27 完成exp_3_3

04.12 完成exp_4_4
## exp4_4的一些大坑
1. cws_op_power_difference.cc 需要注释以下内容才能成功编译tensorflow
  ``` c
    // #include "tensorflow/core/kernels/cwise_op_power_difference_mlu.h"
    #if CAMBRICON_MLU
    // #define REGISTER_MLU(T)                                         \
    //   REGISTER_KERNEL_BUILDER(                                      \
    //       Name("PowerDifference")                                 \
    //           .Device(DEVICE_MLU)                                   \
    //           .TypeConstraint<T>("T"),                              \
    //       MLUPowerDifferenceOp<T>);
    // TF_CALL_MLU_FLOAT_TYPES(REGISTER_MLU);
    #endif  // CAMBRICON_MLU
  ```
  陈云霁视频讲到这部分是第五章实验要用的，所以必须注释，但是教程手册上却没有写
 
2. 运行bash build_tensorflow-v1.10_mlu.sh时，如果出现报错“Server terminated abruptly(erroe code: 14, erroe message: 'Socket closed')”
   则需要加上参数-j 20，限制build进程数量，避免内存不足导致失败
   ```bash
   bash build_tensorflow-v1.10_mlu.sh -j 20
   ``` 
  
## exp5的一些大坑
希冀平台的提交就很玄学，不小心把5-1的结果提交进了5-2，5-2都能满分通过。
1. 提交5-1实验时，可以不提交重新编译的tensorflow（tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl）
2. 提交5-3实验时，需要提交tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl才能通过（我当时是这样）

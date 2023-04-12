# AICS-homework

2023年春季学期 AICS作业存档



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
  

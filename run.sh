python ./code/test.py model=VGG_16_deconv_Fine_grained_DA_test gradient_scalar=-0.01 comment=99.3_vision_SHA \
tar_dataset=SHA_test dataroot_SHA=./data/SHA vision_each_epoch=5000 \
model_for_load=SHA_99.3.pth

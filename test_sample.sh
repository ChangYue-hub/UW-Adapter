python ./test_sample.py \
      --image_path ./images \
      --output_path ./outputs \
      --JTNet_path ./models/others/JTNet.pth \
      --backbone_path ./models/backbone/dpt_swin2_base_384.pt \
      --t_adapter_path ./models/adapter/SwinV2_base/t_adapter.pth \
      --h_adapter_path ./models/adapter/SwinV2_base/h_adapter.pth \
      --backbone swin2b24_384

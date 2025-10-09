时序数据压缩
数据集为CMAPSS FD001
1.使用config_w32.yaml配置
2.运行 data/preprocess_cmapss.py处理数据集
3.运行scripts/assemble_raw_from_txt.py,从 CMAPSS 原始 txt 拼回指定 sensor 的完整原始序列并保存为 npy.
3.运行train.py训练模型
4.运行eval_saved.py,提取训练结果
5.运行scripts/compare_full_raw_vs_rec.py,生成重构数据与原始数据的对比，计算压缩率。

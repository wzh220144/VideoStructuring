# 0.简介
多模态视频标签模型框架

# 1. 代码结构
- configs--------------------# 模型选择，参数文件
- src------------------------# 数据加载/模型相关代码
- scripts--------------------# 数据预处理/训练/测试脚本
- checkpoints----------------# 模型权重/日志
- pretrained-----------------# 预训练模型
- dataset--------------------# 数据集和标签字典
- utils----------------------# 工具脚本
- ReadMe.md

# 2. 环境配置
sudo apt-get update
sudo apt-get install libsndfile1-dev ffmpeg imagemagick nfs-kernel-server
pip install -r requirement.txt

## 2.1 配置 imagemagick
删除或注释配置文件/etc/ImageMagick-6/policy.xml中的:
`<policy domain="path" rights="none" pattern="@*" />`

# 3. 训练流程
## 3.1 数据预处理
根据特定任务准备训练数据，视频/音频特征提取可参考scripts/preprocess.sh
bash scripts/preprocess.sh tagging.txt 0 0 4 1

## 3.2 加载预训练模型参数
参考pretrained目录下说明

## 3.3 启动训练
python scripts/train_tagging.py --config configs/config.ad_content.yaml

## 3.4 训练验证tensorboard曲线
tensorboard --logdir checkpoints --port 8080

# 4. 验证流程
python scripts/eval.py --config configs/config.ad_content.yaml --ckpt_step -1

1. 分别输出多模态融合特征, 视觉特征，音频特征，文本特征的评测指标:
  * Hit@1(模型预测得分最高标签的准确率)
  * PERR(按预测得分大小，取前k个预测输出tag对应的准确率,其中k=该样本gt中包含的标签个数)
  * MAP(mean average precision)
  * [GAP(Global Average Precision)](https://www.kaggle.com/c/youtube8m/overview/evaluation)
2. 输出对每个标签的频次统计和每个标签的ap(average precision) **用于分析每个标签的准确度**
3. 输出各个标签之间的相关性统计矩阵M, $M_{a,B}$即样本标签为a时, 模型预测为B(b1,b2,...)的分布频次统计, 保留前top_k个结果保**用于将相似标签合并，更新标签字典文件**
4. 保存输出文件eval_tag_analysis.txt,  `每行依次表示tag_freq,tag_ap,tag_conf,tag_precision,tag_recall`， 通过scripts/tag_analysis.ipynb对验证结果进行分析

# 5. 测试流程
python scripts/inference.py --model_pb checkpoints/ad_content_form/v1/export/step_7000_0.8217 \
                            --tag_id_file dataset/dict/tag-id-ad_content_b0.txt \
                            --test_dir dataset/looklike_interview \
                            --postfix mp4 \
                            --output ./pred_output.txt \
                            --top_k 5
> 参数说明
```
--model_pb 导出模型pb目录
--tag_id_file 标签字典文件
--test_dir 输入测试文件目录
--output 预测输出标签结果保存文件
--top_k 预测输出标签个数
--postfix 测试文件格式, mp4或者jpg文件
```

# 6. Badcase 分析
## 6.1 预测可视化
python scripts/write_prediction.py --inference_file checkpoints/ad_content_form/v1/inference_result_fusion.txt --sample_num 200 --save_dir temp --tag_id_file dataset/dict/tag-id-ad_content_b0.txt --test_dir dataset/videos/ad_content --gt_file dataset/info_files/ad_content_datafile_b0.txt --postfix mp4

> 参数说明
```
--inference_file 预测输出标签结果保存文件
--sample_num 随机采样可视化sample_num个视频
--gt_file 样本对应gt文件(可选项）
--save_dir 可视化文件保存路径
--filter_tag_name 只可视化带有该标签的样本(可选项）
--tag_id_file 标签字典文件(可选项,当filter_tag_name不为空时需要)
--test_dir 测试文件目录
--postfix 测试文件格式, mp4或者jpg文件
```

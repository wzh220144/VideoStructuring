# **2021年腾讯广告算法大赛baseline**

欢迎各位参赛选手参加腾讯广告算法大赛。本次比赛主要的内容是关注广告视频的解析任务，即利用多模态信息对一段长度为几十秒的广告视频进行时序上的分割与分类。具体来说是要将一段视频稠密地分割为许多语义连续的“幕”（scene），这些幕之间没有交集，并且所有幕的并集是完整的原始视频长度。解析任务除了关注幕的语义边界的准确性，还关注分割出来的每个幕的分类结果。本次比赛标签体系包含三个维度： “呈现形式”，“风格”，“场景”，每一个维度内部都是一个多标签分类任务。选手需要预测出幕的边界和每个幕在三个标签体系内的标签类别。

本次比赛分为两个赛道，其中赛道一是上述完整的广告视频秒级解析任务；赛道二将赛道一简化为一个多标签任务，去除了时间上秒级解析的要求，也就是去除了幕的概念，将一个视频内所有幕的多标签聚合起来（集合的加法）作为整个视频的标签，只要求选手预测整个视频的多标签分类。

针对赛道一，我们选取了一个在电影视频上的多模态scene segmentation方法LGSS [4] 作为本次比赛秒级解析部分的baseline，并且在分割出scene以后，利用一个深度网络在通过NextVlad [8] 聚合出的scene-level多模态特征上预测三个标签体系的多标签类别。LGSS的详细内容请参考文章内容[4] 和[链接](https://github.com/AnyiRao/SceneSeg)，但是请注意电影视频的数据分布与本次比赛使用的广告视频的数据分布可能不同，所以将模型简单迁移过来的结果可能不会很好，选手应该尽量考虑广告视频的数据特点改进这份baseline。

针对赛道二，正如前文所说，赛道二是赛道一的分类任务简化版，因此我们直接使用一个深度网络通过NextVlad聚合出的video-level多模态特征上预测三个标签体系的多标签类别。



## **赛道一  视频广告秒级语义解析**

比赛组织方提供的本赛道的baseline由两个阶段组成，首先使用LGSS进行二分类的幕切割；然后使用幕切割的结果聚合特征，对每个幕进行三个标签层次的多分类。选手也可以尝试开发一阶段的方法同时进行幕切割和幕分类，同时进行分割与分类有可能对于语义理解有利。

### **1.1 环境配置**

```shell
sudo apt-get install libsndfile1-dev ffmpeg
cd MultiModal-Tagging
pip install -r requirement.txt
```

### **1.2 训练流程**

#### **1.2.1 SceneSeg**

**数据说明**:

以下特征都是基于镜头切分的结果进行提取，提取出的特征代表了当前的shot的特征。

├── ../dataset/structruing/structuring_dataset_train_5k 各实验的数据根目录

​		├── aud_feat 音频特征文件。说明：shot-level的音频特征，格式为npy，用于场景分割训练

​		├── place_feat 地点特征文件。说明：shot-level粒度的地点图像特征，格式为npy，用于场景分割训练。

​		├── meta 文件夹。说明：对训练集做验证、测试的的划分。

​		├── labels 标签文件。说明：生成镜头切分的Groundtruth用于做场景分割训练，需注意若视频只生成一个镜头则无法生成相应的label标签（单个镜头无法做场景聚合），该数据只提供4071个label file

​		├──shot_stats, shot_txt 等文件。说明：镜头切分结果相关文件，e.g. 视频镜头文件（shot_txt）等



├──pre **预处理代码**

​		├── ShotDetect **镜头切分，以下会具体说明**

​		├── audio **音频特征提取**，可以修改为任意音频特征提取网络

​		├── place **地点特征提取**，可以修改为任意图片特征提取网络

​		├── gt_generator.py **数据格式转换脚本**，生成可用来直接训练的groudtruth文件。将原始的以时间为单位的幕分割ground-truth近似到最近的镜头分割位置上，生成以镜头为最小单位的0/1序列，即当前镜头切分位置是/否为幕分割位置。这份生成的二分类ground-truth作为LGSS模型的ground-truth进行幕分割的训练。比赛组织方鼓励选手开发自己的幕分割方法，可以丢弃基于镜头切分的ground-truth组织形式，直接以时间/帧为粒度进行切分；也可以参考当前的以镜头为最小单元的方法，其中镜头切分方法可以使用比赛组织方提供的baseline方法，也可以使用任何其他的镜头切分方法。如果继续选择基于镜头切分的方式，本代码可以基于其他的镜头切分结果生成新的二分类ground-truth序列。

├── lgss **模型训练测试代码**

​		├── run.py **代码入口**

​		├── config **配置文件目录**

​		├── src **网络结构等代码**

​		├── utilis **其他被调用的函数**

├── run **模型结果的默认保存文件夹**

------

Baseline使用的镜头切分方法与LGSS使用的方法一致，为非深度学习方法 [5]，精度有限。选手可以尝试使用其他深度学习镜头切分方法，例如TransNet [6]。基于镜头切分的方法可以减少需要处理的序列长度，对于更加抽象层面的语义理解有帮助，但是最终SceneSeg结果依赖镜头切分方法的精度，镜头切分的召回率决定了SceneSeg的上限。原因是基于镜头切分的方法需要将真实ground-truth近似到镜头切分的位置上，如果镜头切分的方法效果较差时，近似误差会很大。不基于镜头切分的方法直接使用帧或clip为最小粒度，上限相比基于镜头切分的方法更高，但是需要处理的序列长度明显大于镜头数，对于时序模型有比较高的要求，并且对于长序列的数据，模型很容易输出过分割 (over-segmentation) 的结果（即输出的幕边界多于实际的幕边界，将一个完整的幕分割成了多个碎片），选手需要重点考虑如何处理这样的问题。选手可以通过在训练集上切分验证集，评估当前使用的镜头切分方法对于真实ground-truth的召回率，综合考虑想要采用的方法。选手也可以结合以上两种方法的优点，形成一个集成两种模型的方法，但是需要注意模型的复杂度，复赛和决赛阶段比赛方会评估选手提交方法的时间复杂度。关于模型的时间复杂度的要求请参考比赛要求。

关于镜头切分或与镜头粒度相似的一般事件边界（event boundary）切分可以参考CVPR 2021 LOVEU Challenge [9]，选手也可以考虑复用这个数据集中的镜头切分部分数据，也可以不拘泥于镜头粒度的切分。总而言之，进行镜头或者类似粒度的切分可以减少后续模型处理的规模，并且由于镜头切换的特征十分明显，可以提供更加精准的边界定位。

此外，SceneSeg阶段比赛方提供的baseline只包含每个镜头内关键帧特征与音频特征；Scene-level multimodal tagging阶段比赛方提供基于ground-truth SceneSeg结果提取的ASR/OCR/视频/关键帧图片的特征，但是用于ASR/OCR的模型比赛方不提供。选手也可以尝试使用公开的或自己开发的ASR/OCR算法抽取更多区间的特征，或者尝试在SceneSeg阶段抽取带有时间戳的ASR/OCR结果，将这部分多模态信息在SceneSeg阶段就加入到模型中。选手使用的ASR/OCR模型需要提供到比赛组织方。

```shell
# 1.1 训练数据准备--镜头切分
cd SceneSeg/pre
python ShotDetect/shotdetect.py 
    --video_dir 
        ../../dataset/videos/train_5k/ 
    --save_dir
        ../../dataset/structuring/structuring_dataset_train_5k/

# 1.2 训练数据准备--特征提取
cd SceneSeg/pre
python place/extract_feat.py --data_root ../../dataset/structuring/structuring_dataset_train_5k/
python audio/extract_feat.py --data_root ../../dataset/structuring/structuring_dataset_train_5k/

# 2. 生成SceneSeg 任务gt
cd SceneSeg/pre
python gt_generator.py 
      --video_dir 
        ../../dataset/videos/train_5k/
      --data_root 
        ../../dataset/structuring/structuring_dataset_train_5k/
      --input_annotation 
        ../../dataset/structuring/GroundTruth/train5k.txt
      
# 3.场景切分, 启动训练
cd  SceneSeg/lgss
python run.py config/train5k.py 

```

#### **1.2.2 MultimodalTagging**

Baseline方法基于SceneSeg模型预测的边界使用NextVlad [8]将shot-level特征聚合为scene-level特征，然后使用一个深度网络进行Multimodal Tagging任务。由于在 MultimodalTagging 阶段我们假设提取出来的scene是一些语义连续的片段，因此在baseline方法在这部分加入了OCR/ASR的识别结果来进一步增强分类效果。基于这样的考虑，baseline方法没有复用在SceneSeg阶段提取出来的特征，而是重新提取了特征。由于特征抽取时间较长，比赛组织方在训练集上提供了基于ground-truth SceneSeg边界的每个幕的特征便于选手训练分类模型。需要注意的是在预测阶段SceneSeg模型输出的scene区间与ground-truth scene区间有可能有数据分布的不一致，因此直接将在ground-truth SceneSeg区间特征上训练的Multimodal Tagging模型用于测试有可能会导致较差的效果。选手需要注意让自己训练出的Multimodal Tagging模型适应前一阶段训练出的SceneSeg结果的数据分布。选手也可以开发一阶段方法，复用特征。此外，基于不能泄露SceneSeg的ground-truth的考虑，在测试集上我们无法提供OCR/ASR结果。尽管baseline的MultimodalTagging模型在训练时会随机丢弃模态，并在测试时能够基于给出的模态做出预测，在测试集上直接丢弃OCR/ASR信息的做法可能会导致较差的结果。选手可以结合比赛组织方给出的视频demo尝试理解广告视频的数据特点，并调整合适的模态之间重要性的权重。



数据说明:

提供数据集压缩文件dataset.zip。场景（scene）标签任务，场景粒度的视频特征

├── ../dataset/structuring/train5k_split_video_feats 各实验的数据根目录

​		├── audio_npy 音频特征文件。说明：以场景粒度提取的音频特征，格式为npy。

​		├── image_jpg 视频抽样图片。说明：以场景粒度采样的图片，格式为jpg。

​		├── text_txt 视频OCR/ASR识别结果。说明：格式为txt文档。

​		├── video_npy 视频特征。说明：以场景粒度提取的视频特征，格式为npy。

------

```shell
cd MultiModal-Tagging
# 1.特征提取(提供预处理好的特征数据 && 源视频文件), 该步骤可忽略
# 2.将解析任务的GT转换成标签任务的GT（info file），比赛有提供，该步骤可忽略
python scripts/preprocess/json2info.py 
    --video_dir ../dataset/videos/train_5k 
    --json_path ../dataset/structuring/GroundTruth/train5k.txt 
    --save_path ../dataset/structuring/GroundTruth/structuring_tagging_info.txt 
    --convert_type structuring 
    --split_video_dir ../dataset/structuring/train5k_split_video/

# 3. 根据文件路径生成训练框架对应GT文件
python scripts/preprocess/generate_datafile.py 
    --info_file  ../dataset/structuring/GroundTruth/structuring_tagging_info.txt 
    --out_file_dir ../dataset/structuring/GroundTruth/datafile/ 
    --tag_dict_path ../dataset/label_id.txt 
    --frame_npy_folder ../dataset/structuring/train5k_split_video_feats/video_npy/ 
    --audio_npy_folder ../dataset/structuring/train5k_split_video_feats/audio_npy/ 
    --text_txt_folder ../dataset/structuring/train5k_split_video_feats/text_txt/ 
    --image_folder ../dataset/structuring/train5k_split_video_feats/image_jpg/

# 4.训练流程
python scripts/train_tagging.py 
    --config configs/config.structuring.5k.yaml

```

#### **1.3 测试流程**

```shell
# 1.镜头切分
cd SceneSeg/pre
python ShotDetect/shotdetect.py 
    --video_dir 
        ../../dataset/videos/test_5k/ 
    --save_dir
        ../../dataset/structuring/structuring_dataset_test_5k/

# 2.特征提取
cd SceneSeg/pre
python place/extract_feat.py 
    --data_root ../../dataset/structuring/structuring_dataset_test_5k/
python audio/extract_feat.py 
    --data_root ../../dataset/structuring/structuring_dataset_test_5k/

# 3.场景切分
cd  SceneSeg/lgss
python run_inference.py 
    --config config/common_test.py 
    --data_root ../../dataset/structuring/structuring_dataset_test_5k/ 
    --model_path ../run/sceneseg_train5k/model_best.pth.tar

# 4. 多模态标签
cd MultiModal-Tagging
python scripts/inference_for_structuring.py 
    --model_pb checkpoints/structuring_tagging_train5k/export/step_8000_0.6149 
    --tag_id_file ../dataset/label_id.txt 
    --test_dir ../dataset/structuring/structuring_dataset_test_5k/scene_video 
    --output_json results/structuring_tagging_5k.json

```

------



## **赛道二  多模态视频广告标签**

Baseline方法在赛题二上的模型与赛题一中MultimodalTagging一致，唯一的区别是数据和标注：赛题一的多模态分类为scene-level，赛题二为video-level。请选手参考本份baseline说明文档赛题一的MultimodalTagging部分。

### **2.1 环境配置**

```shell
sudo apt-get update
sudo apt-get install libsndfile1-dev ffmpeg
cd MultiModal-Tagging
pip install -r requirement.txt
```

### **2.2 训练流程**

数据说明:

提供数据集压缩文件dataset.zip

├── ../dataset/tagging/tagging_dataset_train_5k 各实验的数据根目录

​		├── audio_npy 音频特征文件。说明：以视频粒度提取的音频特征，格式为npy。

​		├── image_jpg 视频抽样图片。说明：以视频粒度提采样的图片，格式为pkl。

​		├── text_txt 视频OCR/ASR识别结果。说明：格式为txt文档。

​		├── video_npy 视频特征。说明：以视频粒度提取的视频特征，格式为npy。



```shell
# 1.特征提取(提供预处理好的特征数据 && 源视频文件), 该步骤可忽略
# 2.将解析任务的GT转换成标签任务的GT（info file），比赛有提供，该步骤可忽略
python scripts/preprocess/json2info.py 
    --video_dir ../dataset/videos/train_5k 
    --json_path ../dataset/structuring/GroundTruth/train5k.txt 
    --save_path ../dataset/tagging/GroundTruth/tagging_info.txt 
    --convert_type tagging 
    
# 3. 根据文件路径生成训练框架对应GT文件
python scripts/preprocess/generate_datafile.py 
    --info_file ../dataset/tagging/GroundTruth/tagging_info.txt
    --out_file_dir ../dataset/tagging/GroundTruth/datafile/
    --tag_dict_path ../dataset/label_id.txt
    --frame_npy_folder ../dataset/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/
    --audio_npy_folder ../dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging/
    --text_txt_folder ../dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/
    --image_folder ../dataset/tagging/tagging_dataset_train_5k/image_jpg/tagging/
# 4. 训练流程
cd MultiModal-Tagging
python scripts/train_tagging.py --config configs/config.tagging.5k.yaml

```

### **2.3 测试流程**

```shell
cd MultiModal-Tagging
# inference，在当前目录下输出符合赛题要求的json格式预测文件
python scripts/inference_for_tagging.py  
    --model_pb checkpoints/tagging5k/export/step_7000_0.7359/ 
    --tag_id_file ../dataset/label_id.txt 
    --test_dir ../dataset/videos/test_5k/ 
    --output_json results/tagging_5k.json
    --load_feat 1 
    --feat_dir ../dataset/tagging/tagging_dataset_test_5k
```

------

## **参考文献**

[1] Hershey, Shawn, et al. "CNN architectures for large-scale audio classification." *2017 ieee international conference on acoustics, speech and signal processing (icassp)*. IEEE, 2017.

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." *2009 IEEE conference on computer vision and pattern recognition*. Ieee, 2009.

[3] He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[4] Rao, Anyi, et al. "A local-to-global approach to multi-modal movie scene segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

[5] Sidiropoulos, Panagiotis, et al. "Temporal video segmentation to scenes using high-level audiovisual features." *IEEE Transactions on Circuits and Systems for Video Technology* 21.8 (2011): 1163-1177.

[6] Souček, Tomáš, Jaroslav Moravec, and Jakub Lokoč. "TransNet: A deep network for fast detection of common shot transitions." *arXiv preprint arXiv:1906.03363* (2019).

[7] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[8] Lin, Rongcheng, Jing Xiao, and Jianping Fan. "Nextvlad: An efficient neural network to aggregate frame-level features for large-scale video classification." *Proceedings of the European Conference on Computer Vision (ECCV) Workshops*. 2018.

[9] Shou, Mike Zheng, et al. "Generic Event Boundary Detection: A Benchmark for Event Segmentation." arXiv preprint arXiv:2101.10511 (2021).

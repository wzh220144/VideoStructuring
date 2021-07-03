experiment_name = "train_transnet_v2_log"
experiment_description = "scene segmentation with all modality"

data_root = '/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2'
video_dir = '/home/tione/notebook/dataset/videos/train_5k_A'
model_path = '/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/model'
shot_frm_path = data_root + "/shot_txt"
shot_num = 6  # even
seq_len = 4  # even
gpus = "0,1"

place_base = 'place_feat'
vit_base = 'vit_feat'
aud_base = 'aud_feat'

dataset = dict(
    name="train",
    mode=['place', 'aud'],
)

model = dict(
    name='LGSS',
    model_mode = 1,
    sim_channel = 512,  # dim of similarity vector
    place = dict(
        feature_size = 2048,
        frame = 4,
        output_dim = 2048,
        group = 16,
        expansion = 2,
        nextvlad_cluster_size = 128,
        ),
    vit_feat_dim = 768,
    act_feat_dim = 512,
    aud_feat_dim = 512,
    aud = dict(cos_channel = 512),
    bidirectional = True,
    lstm_hidden_size = 512,
    ratio = [0.8, 0, 0, 0.4],
    num_layers = 1,
    reduction = 16,
    dropout_ratio = 0.5,
    se_dim = 64,
    )

optim = dict(name = 'Adam',
             setting = dict(lr = 1e-3, weight_decay = 5e-4))
stepper = dict(name='MultiStepLR', setting=dict(milestones = [25], gamma = 0.1))
loss = dict(weight=[0.5, 5])

resume = None
trainFlag = 1
testFlag = 0
batch_size = 64
epochs = 100
logger = dict(log_interval = 100, logs_dir = "/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers = 14, pin_memory = True, drop_last = True)

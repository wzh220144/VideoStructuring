experiment_name = "train_hsv"
experiment_description = "scene segmentation with all modality"
# overall confg
data_root = '/home/tione/notebook/dataset/train_5k_A/shot_hsv'
video_dir = '/home/tione/notebook/dataset/videos/train_5k_A'
model_path = '/home/tione/notebook/dataset/train_5k_A/shot_hsv/model'
shot_frm_path = data_root + "/shot_txt"
shot_num = 6  # even
seq_len = 4  # even
gpus = "0,1"

# dataset settings
dataset = dict(
    name="train",
    mode=['place', 'aud'],
)
# model settings
model = dict(
    name='LGSS',
    sim_channel=512,  # dim of similarity vector
    place_feat_dim=2048,
    cast_feat_dim=512,
    act_feat_dim=512,
    aud_feat_dim=512,
    aud=dict(cos_channel=512),
    bidirectional=True,
    lstm_hidden_size=512,
    ratio=[0.8, 0, 0, 0.4],
    num_layers=2,
    )

# optimizer
optim = dict(name='Adam',
             setting=dict(lr=1e-3, weight_decay=5e-4))
stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[15, 30, 40, 50], gamma=0.1))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = 1
testFlag = 0
batch_size = 64
epochs = 100
logger = dict(log_interval=100, logs_dir="/home/tione/notebook/dataset/train_5k_A/shot_hsv/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=14, pin_memory=True, drop_last=True)

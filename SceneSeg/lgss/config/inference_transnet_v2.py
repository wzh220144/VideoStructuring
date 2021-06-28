experiment_name = "inference_transnet_v2_log"
experiment_description = "scene segmentation using images only"

# overall confg
data_root = '/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2'
video_dir = '/home/tione/notebook/dataset/videos/test_5k_2nd'
model_path = '/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/model'
output_root = '/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2/output'
shot_frm_path = data_root + "/shot_txt"
shot_num = 6
seq_len = 4
gpus = "0,1"

# dataset settings
dataset = dict(
    name="inference",
    mode=['place', 'aud'],
)
# model settings
model = dict(
    name='LGSS',
    # backbone='resnet50',
    place_feat_dim=2048,
    aud_feat_dim=512,
    aud=dict(cos_channel=512),
    fix_resnet=False,
    sim_channel=512,  # dim of similarity vector
    bidirectional=True,
    lstm_hidden_size=512,
    ratio=[0.8,0, 0, 0.2],
    num_layers=2,
    )

# optimizer
optim = dict(name='SGD',
             setting=dict(lr=1e-2, weight_decay=5e-4))
stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[15]))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = False
testFlag = True
batch_size = 64
epochs = 30
logger = dict(log_interval=200, logs_dir="/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=14, pin_memory=True, drop_last=False)

experiment_name = "train_transnet_v2_log"
experiment_description = "scene segmentation with all modality"

data_root = '/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2'
video_dir = '/home/tione/notebook/dataset/videos/train_5k_A'
model_path = '/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/model'

shot_frm_path = data_root + "/shot_txt"
shot_num = 6  # even
seq_len = 10  # even
gpus = "0,1"

use_gpu = 1

model = dict(
        model_mode = 1,
        name = 'LGSS',
        modal = dict(
            place = dict(
                use = 1,
                sim_channel = 512,
                input_dim = 2048,
                lstm_hidden_size = 512,
                hidden_dim = 100,
                loss_weight = 0.4,
                base = 'place_feat',
                bidirectional = True,
                dropout_ratio = 0.5,
                output_dim = 1024,
                reduction = 8,
                num_layers = 2,
                se_dim = 512,
                ),
            vit = dict(
                use = 1,
                sim_channel = 512,
                input_dim = 768,
                lstm_hidden_size = 512,
                hidden_dim = 100,
                loss_weight = 0.2,
                base = 'vit_feat',
                bidirectional = True,
                dropout_ratio = 0.5,
                output_dim = 1024,
                reduction = 8,
                num_layers = 2,
                se_dim = 512,
                ),
            aud = dict(
                use = 1,
                sim_channel = 512,
                input_dim = 512,
                lstm_hidden_size = 512,
                hidden_dim = 100,
                loss_weight = 0.1,
                base = 'aud_feat',
                bidirectional = True,
                dropout_ratio = 0.5,
                output_dim = 1024,
                reduction = 8,
                num_layers = 2,
                se_dim = 512,
                ),
            fusion = dict(
                output_dim = 1024,
                loss_weight = 0.8,
                reduction = 8,
                dropout_ratio = 0.5,
                se_dim = 512,
                )
            )
        )

optim = dict(name = 'Adam', setting = dict(lr = 1e-3, weight_decay = 5e-4))
stepper = dict(name='MultiStepLR', setting=dict(milestones = [25, 50], gamma = 0.1))
loss = dict(weight=[0.5, 5])

threshold = 0.5

resume = None
trainFlag = 1
testFlag = 0
batch_size = 16
epochs = 100
logger = dict(log_interval = 100, logs_dir = "/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers = 14, pin_memory = True, drop_last = True)

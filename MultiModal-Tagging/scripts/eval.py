#coding=utf-8
#Author: jefxiong@tencent.com
#Author: xxx@tencent.com

import sys,os
sys.path.append(os.getcwd())

import os
import time
import argparse
import yaml
import numpy as np
from munch import Munch
import traceback

import tensorflow as tf
from tensorflow import logging

import src.dataloader.dataloader as dataloader
import src.loss as loss_lib
import src.model.models as models
from utils.train_util import get_latest_checkpoint, get_label_name_dict, EvaluationMetrics,FormatEvalInfo, calculate_gap,format_lines


class ModelEval():
    """
    模型验证基类
    """
    def __init__(self, configs, ckpt_step=-1):
        tf.set_random_seed(0)
        self.ckpt_step = ckpt_step
        self.model_config = Munch(configs['ModelConfig'])
        self.optimizer_config = Munch(configs['OptimizerConfig'])
        self.data_config = configs['DatasetConfig']
        self.configs = configs
        self.modal_name_list =[]
        if self.model_config.with_video_head: self.modal_name_list.append('video')
        if self.model_config.with_audio_head: self.modal_name_list.append('audio')
        if self.model_config.with_text_head: self.modal_name_list.append('text')
        if self.model_config.with_image_head: self.modal_name_list.append('image')
          
        with tf.Graph().as_default():
            self.reader = dataloader.Data_Pipeline(config['DatasetConfig'])
            self.model = models.get_instance(self.model_config.model_type, config['ModelConfig'])
        
    def build_eval_graph(self):
        logging.info("built evaluation graph")
        input_name_list = self.reader.dname_string_list #模型输入变量名
        inupt_shape_list = self.reader.data_shape_list  #模型输入shape
        input_dtype_list = self.reader.dtype_list       #模型输入类型
        inputs_dict={}
        val_predictions={}
        loss_dict = {}
        
        with tf.name_scope("eval_input"):
            for input_name,input_shape,input_dtype in zip(input_name_list, inupt_shape_list, input_dtype_list):
                logging.info("input_name: {}, input_shape:{}, input_dtype: {}".format(input_name, input_shape, input_dtype))
                inputs_dict[input_name] = tf.placeholder(shape=[None]+input_shape, dtype=input_dtype, name=input_name) #add batch size dim

            with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
                result = self.model(inputs_dict, is_training=False)
                for task_name in self.reader.label_num_dict: #[tagging, classification]
                    for modal_name in ['fusion'] + self.modal_name_list:
                        val_predictions[task_name+'_output_'+modal_name] = result[task_name+"_output_"+modal_name]["predictions"]
                    loss_fn_dict={}
                    for task_name, task_loss_type in self.optimizer_config.loss_type_dict.items():
                        loss_fn = loss_lib.get_instance(task_loss_type, paramters_dict={}) #TODO(jefxiong, 支持损失函数的额外参数输入)
                        loss_fn_dict[task_name] = loss_fn
                    loss_dict.update(self.model.build_loss(inputs_dict, result, loss_fn_dict))
        return inputs_dict, val_predictions, loss_dict
    
    def evaluate(self):
        pass

    
class TaggingModelEval(ModelEval):
    def __init__(self, configs, ckpt_step=-1):
        super(TaggingModelEval,self).__init__(configs, ckpt_step=ckpt_step)
        tag_id_file_path = self.configs['DatasetConfig']['preprocess_config']['label'][0]['extra_args']['index_dict']
        self.label_name_dict = get_label_name_dict(tag_id_file_path, None)
        
    def evaluate(self):
        inputs_dict, val_predictions, loss_dict = self.build_eval_graph()
        saver = tf.train.Saver(tf.global_variables())        
        tagging_aux_evl_metrics = [EvaluationMetrics(self.reader.label_num_dict['tagging'], 20, False) 
                           for i in range(len(val_predictions)-1)]
        tagging_aux_evl_metrics.append(EvaluationMetrics(self.reader.label_num_dict['tagging'], 20, True))
        
        #init Eval Metrics
        inference_result_file=[]
        module_list = self.modal_name_list+['fusion']
        for i in range(len(tagging_aux_evl_metrics)):
            tagging_aux_evl_metrics[i].clear()
            f = open(self.optimizer_config.train_dir+ '/' + "inference_result_"+module_list[i]+'.txt', 'w')
            inference_result_file.append(f)
        
        
        with tf.Session() as sess:
            latest_checkpoint = get_latest_checkpoint(self.optimizer_config.train_dir) if self.ckpt_step==-1 else \
                                    os.path.join(self.optimizer_config.train_dir, "model.ckpt-{}".format(self.ckpt_step))
            if latest_checkpoint:
                logging.info("Loading checkpoint for eval: " + latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]
            else:
                logging.info("No checkpoint file found.")
                return global_step_val
            sess.run([tf.local_variables_initializer()])
            
            try:
                examples_processed = 0
                valid_sample_generator_dict =  self.reader.get_valid_sample_generator_dict()
                for source_name, generator in valid_sample_generator_dict.items(): 
                    for sample in generator:
                        batch_start_time = time.time()
                        feed_dict_data = {}
                        for input_name in self.reader.dname_string_list:
                            feed_dict_data[inputs_dict[input_name]] = sample[input_name]
                        inputs_dict_eval, val_predictions_eval, loss_dict_eval = sess.run([inputs_dict, val_predictions, loss_dict], feed_dict=feed_dict_data)
                        seconds_per_batch = time.time() - batch_start_time
                        example_per_second = self.reader.batch_size / seconds_per_batch
                        examples_processed += self.reader.batch_size
                        
                        for index, modal_name in enumerate(self.modal_name_list+['fusion']):
                            pred = val_predictions_eval['tagging_output_'+modal_name]
                            val_label = sample['tagging']
                            gap = calculate_gap(pred, val_label)
                            iteration_info_dict = tagging_aux_evl_metrics[index].accumulate(pred, val_label, loss_dict_eval['tagging_loss_'+modal_name])
                            iteration_info_dict['GAP'] = gap
                            iteration_info_dict["examples_per_second"] = example_per_second
                            iterinfo = "|".join(["{}: {:.3f}".format(k,v) for k,v in iteration_info_dict.items()])
                            logging.info("%s examples_processed: %d | %s", modal_name, examples_processed, iterinfo)

                            for line in format_lines(sample['idx'], pred, 20, self.label_name_dict):
                              inference_result_file[index].write(line)
                            inference_result_file[index].flush()
                            
                logging.info("Done with batched inference. Now calculating global performance metrics.")
                                    
                for index, modal_name in enumerate(self.modal_name_list+['fusion']):
                    epoch_info_dict = tagging_aux_evl_metrics[index].get()
                    epoch_info_dict["epoch_id"] = global_step_val
                    epochinfo = FormatEvalInfo(None, global_step_val, epoch_info_dict, AddSummary=False)
                    logging.info("-"*50+ "  {} prediction Metric  ".format(modal_name)+"-"*50)
                    logging.info(epochinfo)
                    tagging_aux_evl_metrics[index].clear()
                
                # logging and write class MAP into file
                sort_index = np.argsort(epoch_info_dict['num'])[::-1]
                with open(self.optimizer_config.train_dir+'/eval_tag_analysis.txt', 'w') as f_corr:
                    top_k = tagging_aux_evl_metrics[-1].top_k
                    for i in sort_index:
                        tag = self.label_name_dict.get(i, None)
                        row_tag_corr = epoch_info_dict['tag_correlation'][i]
                        row_tag_freq = int(epoch_info_dict['num'][i])
                        row_tag_ap   = epoch_info_dict['aps'][i]
                        row_tag_conf = epoch_info_dict['tag_confidence'][i]
                        row_tag_precision = epoch_info_dict['tag_precision'][i]
                        row_tag_recall = epoch_info_dict['tag_recall'][i]
                        row_tag_info = "{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(i, tag,
                                                                    row_tag_freq,row_tag_ap,row_tag_conf,
                                                                    row_tag_precision, row_tag_recall)
                        correlated_tag_idx = np.argsort(row_tag_corr)[::-1][:top_k]
                        correlated_tag_name = ['\t'+ self.label_name_dict.get(tag_idx, "NULL") + \
                                             ": %d"%row_tag_corr[tag_idx] for tag_idx in correlated_tag_idx]
                        f_corr.write(row_tag_info+'\n'+'\n'.join(correlated_tag_name)+'\n')
                logging.info("writing eval_tag_analysis to {}".format(self.optimizer_config.train_dir+"/eval_tag_analysis.txt"))
                
            except Exception as e:
                logging.info(traceback.format_exc())

        
if __name__ == "__main__":
    logging.set_verbosity(tf.logging.INFO)
    print("tensorflow version: %s" % tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/config.meishi.yaml',type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--ckpt_step',default=-1,type=int)
    args = parser.parse_args()
    config = yaml.load(open(args.config))
    
    tagging_eval = TaggingModelEval(config, args.ckpt_step)
    tagging_eval.evaluate()

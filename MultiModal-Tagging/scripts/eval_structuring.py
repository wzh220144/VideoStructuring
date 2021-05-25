import json
import traceback
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    #print("recall", mrec)
    #print("preciosn", mprec)
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


class ANETdetection(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 tag_id_dict=None, ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        # self.check_status = check_status
        self.blocked_videos = list()
        self.tag_id_dict = {}
        for line in open(tag_id_dict, "r"):
            tag, idx = line.strip().split()
            self.tag_id_dict[tag] = idx
        # print(self.tag_id_dict)
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # print(data)
        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data.items():
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                for label in ann['labels']:
                    '''
                    if label not in activity_index:
                        activity_index[label] = cidx
                        cidx += 1
                    '''
                    if label not in self.tag_id_dict:
                        print("{} not in tag_id_dict".format(label))
                        continue
                    video_lst.append(videoid)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    label_lst.append(self.tag_id_dict[label])
        
        # print({'video-id': video_lst, 't-start': t_start_lst, 't-end': t_end_lst, 'label': label_lst})
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        # print(activity_index)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Pa:rameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        '''
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')
        '''
        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data.items():
            if videoid in self.blocked_videos:
                continue
            for result in v["result"]:
                for i in range(len(result['labels'])):
                    label = self.tag_id_dict[result['labels'][i]]
                    video_lst.append(videoid)
                    t_start_lst.append(float(result['segment'][0]))
                    t_end_lst.append(float(result['segment'][1]))
                    label_lst.append(label)
                    score_lst.append(result['scores'][i])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        # print('!'*50)
        # print(prediction_by_label, label_name, cidx)
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print('Warning: No predictions of label {} were provdied.'.format(label_name))
            return pd.DataFrame()

    def _get_gt_with_label(self, gt_by_label, cidx):
        try:
            return gt_by_label.get_group(cidx).reset_index(drop=True)
        except:
            return pd.DataFrame({"video-id":[], "t-start":[], "t-end":[], "label":[]})
        

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.tag_id_dict)))
        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')
        results = Parallel(n_jobs=64)(
                    delayed(compute_average_precision_detection)(
                        ground_truth=self._get_gt_with_label(ground_truth_by_label, cidx),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.tag_id_dict.items())
        
        for i, cidx in enumerate(self.tag_id_dict.values()):
            ap[:,int(cidx)] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()
        print("mAP at different tiou(0.5-0.95): ", self.mAP)
        #for t in range(len(self.ap[0])):
        #    print(self.ap[:, t])
        return self.average_mAP

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth)) #######
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    
    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            #print(this_pred['video-id'] in ground_truth_gbvn.all().index, this_pred['video-id'])
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
            #if this_pred['video-id'] in list(ground_truth_gbvn.all().index):
            #    print(this_pred['video-id'] , list(ground_truth_gbvn.all().index))
        except Exception as e:
            #print(this_pred['video-id'] in ground_truth_gbvn.all().index, this_pred['video-id'])
            fp[:, idx] = 1
            continue



        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1


    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    
    if len(ground_truth)==0:
        recall_cumsum = np.ones(tp_cumsum.shape)
    else:
        recall_cumsum = tp_cumsum / npos
    
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default="train799.json", type=str)
    parser.add_argument('--pred', default="structuring_train799_pred.json", type=str)
    parser.add_argument('--tag_id_dict', default="tag-id-tagging.txt", type=str)
    args = parser.parse_args()

    anet = ANETdetection(ground_truth_filename = args.gt, prediction_filename = args.pred, tag_id_dict = args.tag_id_dict)
    average_mAP = anet.evaluate()
    print('Average-mAP: {}'.format(average_mAP))
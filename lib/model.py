#-*- coding:utf-8 -*-
import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        self.data_loader_train = build_data_loader(cfg, split="train", is_distributed=distributed)
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed)

        cfg.DATASET.IND_TO_OBJECT = self.data_loader_train.dataset.ind_to_classes
        cfg.DATASET.IND_TO_PREDICATE = self.data_loader_train.dataset.ind_to_predicates

        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Train data size: {}".format(len(self.data_loader_train.dataset)))
        logger.info("Test data size: {}".format(len(self.data_loader_test.dataset)))

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()

        num_error = 0
        for i, data in enumerate(self.data_loader_train, start_iter):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            try:
                loss_dict = self.scene_parser(imgs, targets)
            except ValueError:
                num_error += 1
                print('\n** zero bbox {}th, iter_number:{}\n'.format(num_error, i))
                continue
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.scene_parser.parameters(), 5.)
            self.sp_optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions):
        visualize_folder = "visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            img = imgs.tensors[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_ids[i])), result)

    def test(self, timer=None, visualize=False):
        # 기존 코드:
        #  - 13000iter쯤에 omm 오류 발생
        #  - 주 원인 : iter마다 output을 targets_dict, results_dict, results_pred_dict, reg_recalls에 저장
        #    => 평가가 종료된 뒤에 모든 결과를 이용하여 OD, SGG score 계산
        # 현재 수정한 코드:
        #  - results_pred_dict, targets_dict, reg_recalls를 제거하고 SG_evalutator를 새로 구현 (100개 결과를 대상으로 기존 score와 비교했을 때 동일함)
        #    => 매 iter마다 SGG score를 계산하도록함 (OD는 그대로)
        #  - 대신 test시 batch size는 반드시 1로해야함
        #  - 또 메모리 오류('killed')가 발생하면 results_dict를 손봐야할듯
        #  - visualize시, GT/PRED triple 출력되도록 기능 추가함
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        #targets_dict = {}
        results_dict = {}  # 메모리 오류나면 얘도 안쓰는 방법으로 바꿔야함
        # if self.cfg.MODEL.RELATION_ON:
        #     results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        #reg_recalls = []
        print('self.data_loader_test#:', len(self.data_loader_test))
        sg_evaluator = SG_evaluator(self.data_loader_test.dataset)
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            if i != 0 and i % 5000 == 0: ##
                if self.cfg.MODEL.RELATION_ON:
                    sg_evaluator.print()
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)

                if self.cfg.MODEL.RELATION_ON:
                    #print(output) ##
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                #ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                #reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                #reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                targets = [t.to(cpu_device) for t in targets] ##
                torch.cuda.empty_cache() ##
                if visualize:
                    #output : [BoxList(num_boxes=X, image_width=1024, image_height=768, mode=xyxy)]
                    # output[0].bbox => bounding boxes [[xmin, ymin, xmax, ymax],]
                    # output[0].get_field('labels') => labels [int,]
                    # output[0].extra_fields.keys() => {'labels', 'scores', 'logits', 'features'}

                    #output_pred : [BoxPairList(num_boxes=X*(X-1), image_width=1024, image_height=768, mode=xyxy)]
                    # output_pred[0].bbox => bbox pairs [[xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2],]
                    # output_pred[0].get_field('idx_pairs') => box idxs [[0, 1], [0, 2], ...]
                    # output_pred[0].get_field('scores') => score#:51 [[...(#51)], [...(#51)], ]
                    # output_pred[0].extra_fields.keys() => {'idx_pairs', 'scores'}

                    #targets : [BoxPairList(num_boxes=Y, image_width=1024, image_height=768, mode=xyxy)]
                    # targets[0].get_field('labels') => obj_labels [...]
                    # targets[0].get_field('pred_labels') => rel_matrix (Y*Y) [[0, 0, 0, 43, 0, ...], ...]
                    # targets[0].get_field('relation_labels') => triples [[sub_id, rel_label, obj_id], ]
                    # targets[0].extra_fields.keys() => {'labels', 'pred_labels', 'relation_labels'}

                    # self.data_loader_test.dataset.ind_to_classes
                    # self.data_loader_test.dataset.ind_to_predicates
                    #import pdb
                    #pdb.set_trace()
                    print('')
                    print(f'*img_id: {image_ids}')
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output)

                    try:
                    ##
                        obj_i2s = self.data_loader_test.dataset.ind_to_classes
                        rel_i2s = self.data_loader_test.dataset.ind_to_predicates
                        print('* GT *')
                        for _i, (sub_id, obj_id, rel_label) in enumerate(targets[0].get_field('relation_labels')):
                            sub_label = targets[0].get_field('labels')[sub_id]
                            obj_label = targets[0].get_field('labels')[obj_id]
                            print(f'  {_i+1}. {obj_i2s[sub_label]} - {rel_i2s[rel_label]} - {obj_i2s[obj_label]}')

                        print('\n* PRED *')
                        obj_labels = output[0].get_field('labels')
                        rel_scores = np.array(output_pred[0].get_field('scores')) # (Y, 51)
                        rel_labels = rel_scores.argmax(-1) # (Y)
                        rel_idxs = np.logical_and(rel_labels != 0, rel_scores.max(-1) > 0.2)  # not background and threshold
                        bbox_id_pairs = np.array(output_pred[0].get_field('idx_pairs'))

                        for _i, ((sub_id, obj_id), rel_label) in enumerate(zip(bbox_id_pairs[rel_idxs], rel_labels[rel_idxs])):
                            print(f'  {_i+1}. {obj_i2s[obj_labels[sub_id]]} - {rel_i2s[rel_label]} - {obj_i2s[obj_labels[obj_id]]}')
                        print('')
                    ##
                    except Exception as e:
                        print(e)
                        import pdb
                        pdb.set_trace()
                        print('')


            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            # targets_dict.update(
            #     {img_id: target for img_id, target in zip(image_ids, targets)}
            # )
            if self.cfg.MODEL.RELATION_ON:
                # results_pred_dict.update(
                #     {img_id: result for img_id, result in zip(image_ids, output_pred)}
                # )
                sg_evaluator.update_result(image_ids[0], output[0], output_pred[0])  ##
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
            #print(results_dict, targets_dict, results_pred_dict)

        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        #predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)
        predictions = list(results_dict.values())
        # if self.cfg.MODEL.RELATION_ON:
        #     #predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        #     predictions_pred = list(results_pred_dict.values())

        ## for test
        #print('results_dict\n -', results_dict) # (iter,)
        #print('predictions\n -', predictions) # (iter,)
        #print('results_pred_dict\n -', results_pred_dict) # (iter,)
        #print('predictions_pred\n -', predictions_pred) # (iter,)
        #import pdb
        #pdb.set_trace()
        #print('')
        #exit()
        ##
        if not is_main_process():
            return

        if self.cfg.MODEL.RELATION_ON:
            # eval_sg_results = evaluate_sg(dataset=self.data_loader_test.dataset,
            #                 predictions=predictions,
            #                 predictions_pred=predictions_pred,
            #                 output_folder=output_folder,
            #                 **extra_args)

            sg_evaluator.print()
        '''
        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            # if self.cfg.MODEL.RELATION_ON:
            #     torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))

        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)
        '''



from .data.evaluation.sg.sg_eval import evaluate as sg_eval_func
from .data.evaluation.sg.evaluator import BasicSceneGraphEvaluator

class SG_evaluator:
    # 메모리 문제 해결을 위해 새로 구현함 (OD는 기존 방법 그대로 사용)

    def __init__(self, dataset):
        self.dataset = dataset
        self.evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

        self.top_Ns = [20, 50, 100]
        self.modes = ["sgdet"]
        self.total_result_dict = {}
        self.result_dict = {}
        for mode in self.modes:
            self.result_dict[mode + '_recall'] = {20:[], 50:[], 100:[]}
            self.total_result_dict[mode + '_recall'] = {20: [0., 0], 50: [0., 0], 100: [0., 0]}

        self.logger = logging.getLogger("scene_graph_generation.inference")


    def update_result(self, image_id, prediction, prediction_pred):
        for mode in self.modes:
            #for image_id, (prediction, prediction_pred) in enumerate(zip(predictions, predictions_pred)):
            img_info = self.dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]

            gt_boxlist = self.dataset.get_groundtruth(image_id)

            gt_entry = {
                'gt_classes': gt_boxlist.get_field("labels").numpy(),
                'gt_relations': gt_boxlist.get_field("relation_labels").numpy().astype(int),
                'gt_boxes': gt_boxlist.bbox.numpy(),
            }

            # import pdb; pdb.set_trace()
            prediction = prediction.resize((image_width, image_height))
            obj_scores = prediction.get_field("scores").numpy()
            all_rels = prediction_pred.get_field("idx_pairs").numpy()
            fp_pred = prediction_pred.get_field("scores").numpy()
            # multiplier = np.ones((obj_scores.shape[0], obj_scores.shape[0]))
            # np.fill_diagonal(multiplier, 0)
            # fp_pred = fp_pred * multiplier.reshape(obj_scores.shape[0] * (obj_scores.shape[0] - 1), 1)
            scores = np.column_stack((
                obj_scores[all_rels[:,0]],
                obj_scores[all_rels[:,1]],
                fp_pred.max(1)
            )).prod(1)
            sorted_inds = np.argsort(-scores)
            sorted_inds = sorted_inds[scores[sorted_inds] > 0] #[:100]

            pred_entry = {
                'pred_boxes': prediction.bbox.numpy(),
                'pred_classes': prediction.get_field("labels").numpy(),
                'obj_scores': prediction.get_field("scores").numpy(),
                'pred_rel_inds': all_rels[sorted_inds],
                'rel_scores': fp_pred[sorted_inds],
            }

            self.evaluator[mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )

            sg_eval_func(gt_boxlist.get_field("labels"), gt_boxlist.bbox, gt_boxlist.get_field("pred_labels"),
                    prediction.bbox, prediction.get_field("scores"), prediction.get_field("labels"),
                    prediction_pred.get_field("idx_pairs"), prediction_pred.get_field("scores"),
                         self.top_Ns, self.result_dict, mode)

            for mode in self.modes:
                key = mode + '_recall'

                for top_N in self.top_Ns:
                    if len(self.result_dict[key][top_N]) != 1:
                        import pdb
                        pdb.set_trace()
                        print('')
                    self.total_result_dict[key][top_N][0] += np.array(self.result_dict[key][top_N][0])
                    self.total_result_dict[key][top_N][1] += 1
                    self.result_dict[key][top_N] = []  ## reset

    def print(self):
        self.logger.info("performing scene graph evaluation.")

        for mode in self.modes:
            key = mode + '_recall'

            self.evaluator[mode].print_stats(self.logger)
            self.logger.info('=====================' + mode + '(IMP)' + '=========================')
            self.logger.info("{}-recall@20: {}".format(mode, self.total_result_dict[key][20][0] / self.total_result_dict[key][20][1]))
            self.logger.info("{}-recall@50: {}".format(mode, self.total_result_dict[key][50][0] / self.total_result_dict[key][50][1]))
            self.logger.info("{}-recall@100: {}".format(mode, self.total_result_dict[key][100][0] / self.total_result_dict[key][100][1]))

def build_model(cfg, arguments, local_rank, distributed):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed)

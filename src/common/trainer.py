import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger
from .min_norm_solvers import MinNormSolver
from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator
from geomloss import SamplesLoss


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, student_config, teacher_config, student_model, teacher_model):
        self.student_config = student_config
        self.teacher_config = teacher_config
        self.student_model = student_model
        self.teacher_model = teacher_model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, student_config, teacher_config, student_model, teacher_model):
        super().__init__(student_config, teacher_config, student_model, teacher_model)

        self.logger = getLogger()
        self.start_epoch = 0
        self.cur_step = 0

        self.student_learner = student_config['learner']
        self.student_learning_rate = student_config['learning_rate']
        self.student_epochs = student_config['epochs']
        self.student_eval_step = min(student_config['eval_step'], self.student_epochs)
        self.student_stopping_step = student_config['stopping_step']
        self.student_clip_grad_norm = student_config['clip_grad_norm']
        self.student_valid_metric = student_config['valid_metric'].lower()
        self.student_valid_metric_bigger = student_config['valid_metric_bigger']
        self.student_test_batch_size = student_config['eval_batch_size']
        self.student_device = student_config['device']
        self.student_weight_decay = 0.0
        if student_config['weight_decay'] is not None:
            wd = student_config['weight_decay']
            self.student_weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.student_req_training = student_config['req_training']



        tmp_dd = {}
        for j, k in list(itertools.product(student_config['metrics'], student_config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.student_best_valid_score = -1
        self.student_best_valid_result = tmp_dd
        self.student_best_test_upon_valid = tmp_dd
        self.student_train_loss_dict = dict()
        self.student_optimizer = self._build_optimizer(student_config)

        lr_scheduler = student_config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.student_optimizer, lr_lambda=fac)
        self.student_lr_scheduler = scheduler

        self.student_eval_type = student_config['eval_type']
        self.student_evaluator = TopKEvaluator(student_config)


        self.teacher_learner = teacher_config['learner']
        self.teacher_learning_rate = teacher_config['learning_rate']
        self.teacher_epochs = teacher_config['epochs']
        self.teacher_eval_step = min(teacher_config['eval_step'], self.teacher_epochs)
        self.teacher_stopping_step = teacher_config['stopping_step']
        self.teacher_clip_grad_norm = teacher_config['clip_grad_norm']
        self.teacher_valid_metric = teacher_config['valid_metric'].lower()
        self.teacher_valid_metric_bigger = teacher_config['valid_metric_bigger']
        self.teacher_test_batch_size = teacher_config['eval_batch_size']
        self.teacher_device = teacher_config['device']
        self.teacher_weight_decay = 0.0
        if teacher_config['weight_decay'] is not None:
            wd = teacher_config['weight_decay']
            self.teacher_weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.teacher_req_training = teacher_config['req_training']

        tmp_dd = {}
        for j, k in list(itertools.product(teacher_config['metrics'], teacher_config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.teacher_best_valid_score = -1
        self.teacher_best_valid_result = tmp_dd
        self.teacher_best_test_upon_valid = tmp_dd
        self.teacher_train_loss_dict = dict()
        self.teacher_optimizer = self._build_teacher_optimizer(teacher_config)

        lr_scheduler = teacher_config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.teacher_optimizer, lr_lambda=fac)
        self.teacher_lr_scheduler = scheduler

        self.teacher_eval_type = teacher_config['eval_type']
        self.teacher_evaluator = TopKEvaluator(teacher_config)




    def _build_optimizer(self, config):
        r"""Init the Optimizer for student model

        Args:
            config (dict): Configuration for the student model

        Returns:
            torch.optim: the optimizer for the student model
        """
        if config['learner'].lower() == 'adam':
            optimizer = optim.Adam(self.student_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(self.student_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(self.student_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.student_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.student_model.parameters(), lr=config['learning_rate'])
        return optimizer
    
    def _build_teacher_optimizer(self, config):
        r"""Init the Optimizer for teacher model

        Args:
            config (dict): Configuration for the teacher model

        Returns:
            torch.optim: the optimizer for the teacher model
        """
        if config['learner'].lower() == 'adam':
            optimizer = optim.Adam(self.teacher_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(self.teacher_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(self.teacher_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.teacher_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.teacher_model.parameters(), lr=config['learning_rate'])
        return optimizer
    
        
    def calculate_kl_divergence(self, student_scores, teacher_scores):
        kl_div = F.kl_div(F.log_softmax(student_scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean')
        return kl_div

    def calculate_sinkhorn_distance(self, student_scores, teacher_scores, blur=0.1):
        return SamplesLoss("sinkhorn", p=1, blur=blur, backend="tensorized")(
            student_scores.unsqueeze(0), teacher_scores.unsqueeze(0)
        )

    

    def _train_epoch(self, student_train_data, teacher_train_data, epoch_idx, student_model=None, teacher_model=None, loss_func=None, teacher_iteration_count=0):

        if self.teacher_model is not None:
            if not self.teacher_req_training:
                return 0.0, []

            self.teacher_model.train()
            teacher_loss_func = self.teacher_model.calculate_loss
            total_teacher_loss = 0.0  

            for batch_idx, teacher_interaction in enumerate(teacher_train_data):
                self.teacher_optimizer.zero_grad()
                teacher_losses = teacher_loss_func(teacher_interaction)

                if isinstance(teacher_losses, tuple):
                    teacher_loss = sum(teacher_losses)
                else:
                    teacher_loss = teacher_losses

                if torch.isnan(teacher_loss):  
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return torch.tensor(0.0), []

                teacher_loss.backward()

                if self.teacher_clip_grad_norm:
                    clip_grad_norm_(self.teacher_model.parameters(), **self.clip_grad_norm)
                
          
                self.teacher_optimizer.step()
                total_teacher_loss += teacher_loss.item()  



        if not self.student_req_training:
            return 0.0, []

        self.student_model.train()
        self.teacher_model.eval()
        loss_func =  self.student_model.calculate_loss
        total_student_loss = 0.0  

        for batch_idx, student_interaction in enumerate(student_train_data):
            self.student_optimizer.zero_grad()
            student_losses = loss_func(student_interaction)
            student_scores = self.student_model.full_sort_predict(student_interaction)

            if isinstance(student_losses, tuple):
                student_loss = sum(student_losses)
            else:
                student_loss = student_losses

            if torch.isnan(student_loss):  
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return torch.tensor(0.0), []
            
            with torch.no_grad():
                teacher_scores = self.teacher_model.full_sort_predict(student_interaction)
                 
            distillation_loss = self.calculate_kl_divergence(student_scores, teacher_scores)
            # distillation_loss = self.calculate_sinkhorn_distance(student_scores, teacher_scores)
            distillation_loss = torch.squeeze(distillation_loss) 
            # print(f"distillation_loss: {distillation_loss}")
            # print(f"student_loss: {student_loss}")

            losses = [student_loss, distillation_loss]
            grads = {}

            for i, loss in enumerate(losses):
                if loss != 0.0:
                    loss.backward(retain_graph=True) 
                    grads[i] = [param.grad.clone() for param in self.student_model.parameters() if param.grad is not None]

            self.student_optimizer.zero_grad()  #

            sol, _ = MinNormSolver.find_min_norm_element(list(grads.values()))
            w_student, w_distillation = sol

            w_student, w_distillation = max(w_student, 0.5), max(w_distillation, 0.0001)
            total_loss = w_student * student_loss + w_distillation * distillation_loss
            total_loss.backward()
        
            if self.student_clip_grad_norm:
                clip_grad_norm_(self.student_model.parameters(), **self.clip_grad_norm)
            
            self.student_optimizer.step()
            total_student_loss += total_loss.item()

        return total_student_loss, total_teacher_loss



    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.student_valid_metric] if self.student_valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output += ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, student_train_data, teacher_train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the student_train_data and the valid data.

        Args:
            student_train_data (DataLoader): the student train data
            teacher_train_data (DataLoader): the teacher train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                            If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
            (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        teacher_iteration_count = 0
        
        
        for epoch_idx in range(self.start_epoch, self.student_epochs):  
            # train
            training_start_time = time()
            self.student_model.pre_epoch_processing()
            self.teacher_model.pre_epoch_processing()  
            student_train_loss, teacher_train_loss = self._train_epoch(student_train_data, teacher_train_data, epoch_idx, self.student_model, self.teacher_model, teacher_iteration_count)
            if torch.is_tensor(student_train_loss) or torch.is_tensor(teacher_train_loss):
                # get nan loss
                break
            self.student_lr_scheduler.step()
            self.teacher_lr_scheduler.step()

            self.student_train_loss_dict[epoch_idx] = sum(student_train_loss) if isinstance(student_train_loss, tuple) else student_train_loss
            self.teacher_train_loss_dict[epoch_idx] = sum(teacher_train_loss) if isinstance(teacher_train_loss, tuple) else teacher_train_loss
            training_end_time = time()
            student_train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, student_train_loss)
            teacher_train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, teacher_train_loss)
            post_info = self.student_model.post_epoch_processing()
            if verbose:
                self.logger.info(student_train_loss_output)
                self.logger.info(teacher_train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)


            # eval: To ensure the test result is the best model under validation data, set self.student_eval_step == 1
            if (epoch_idx + 1) % self.student_eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.student_best_valid_score, self.student_cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.student_best_valid_score, self.cur_step,
                    max_step=self.student_stopping_step, bigger=self.student_valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.student_config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.student_best_valid_result = valid_result
                    self.student_best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                (epoch_idx - self.student_cur_step * self.student_eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
            
        return self.student_best_valid_score, self.student_best_valid_result, self.student_best_test_upon_valid
    
    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.student_model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.student_model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.student_config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.student_evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(student_model, teacher_model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    student_config = Config(student_model, dataset, config_dict, mg)
    teacher_config = Config(teacher_model, dataset, config_dict, mg)
    
    init_logger(student_config)
    init_logger(teacher_config)
    student_logger = getLogger()
    teacher_logger = getLogger()
    
    # Print config info
    student_logger.info('██Server: \t' + platform.node())
    student_logger.info('██Dir: \t' + os.getcwd() + '\n')
    student_logger.info(student_config)
    teacher_logger.info('██Server: \t' + platform.node())
    teacher_logger.info('██Dir: \t' + os.getcwd() + '\n')
    teacher_logger.info(teacher_config)

    # Load data
    dataset = RecDataset(student_config)  # Assuming both student and teacher use the same dataset
    # Print dataset statistics
    student_logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    student_logger.info('\n====Training====\n' + str(train_dataset))
    student_logger.info('\n====Validation====\n' + str(valid_dataset))
    student_logger.info('\n====Testing====\n' + str(test_dataset))

    # Wrap into dataloader
    student_train_data = TrainDataLoader(student_config, train_dataset, batch_size=student_config['train_batch_size'], shuffle=True)
    teacher_train_data = TrainDataLoader(teacher_config, train_dataset, batch_size=teacher_config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(student_config, valid_dataset, additional_dataset=train_dataset, batch_size=student_config['eval_batch_size']),
        EvalDataLoader(student_config, test_dataset, additional_dataset=train_dataset, batch_size=student_config['eval_batch_size']))

    ############ Dataset loaded, run model
    hyper_ret = []
    val_metric = student_config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    student_logger.info('\n\n=================================\n\n')

    # Hyper-parameters
    hyper_ls = []
    if "seed" not in student_config['hyper_parameters']:
        student_config['hyper_parameters'] = ['seed'] + student_config['hyper_parameters']
    for i in student_config['hyper_parameters']:
        hyper_ls.append(student_config[i] or [None])
    # Combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # Random seed reset
        for j, k in zip(student_config['hyper_parameters'], hyper_tuple):
            student_config[j] = k
        init_seed(student_config['seed'])
        
        student_logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, student_config['hyper_parameters'], hyper_tuple))

        # Set random state of dataloader
        student_train_data.pretrain_setup()
        teacher_train_data.pretrain_setup()
        # Model loading and initialization
        student_model = get_model(student_config['model'])(student_config, student_train_data).to(student_config['device'])
        teacher_model = get_model(teacher_config['model'])(teacher_config, teacher_train_data).to(teacher_config['device'])
        student_logger.info(student_model)
        teacher_logger.info(teacher_model)
       
        # Trainer loading and initialization
        trainer = get_trainer()(student_config, teacher_config, student_model, teacher_model)
        # Model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(student_train_data, teacher_train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # Save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        student_logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        student_logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        student_logger.info('████Current BEST████:\nParameters: {}={},\n'
                            'Valid: {},\nTest: {}\n\n\n'.format(student_config['hyper_parameters'],
                hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # Log info
    student_logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        student_logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(student_config['hyper_parameters'],
                                                                                          p, dict2str(k), dict2str(v)))

    student_logger.info('\n\n█████████████ BEST ████████████████')
    student_logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(student_config['hyper_parameters'],
                                                                       hyper_ret[best_test_idx][0],
                                                                       dict2str(hyper_ret[best_test_idx][1]),
                                                                       dict2str(hyper_ret[best_test_idx][2])))

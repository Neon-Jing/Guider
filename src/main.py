# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_model', '-sm', type=str, default='SELFCFED_LGN', help='name of the student model')
    parser.add_argument('--teacher_model', '-tm', type=str, default='TeacherModel', help='name of the teacher model')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(student_model=args.student_model, teacher_model=args.teacher_model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)

#!/usr/local/bin/python3
# Copyright (c) 2022 Robert Bosch GmbH Copyright holder of the paper "Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain" accepted at ECCV 2022.
# All rights reserved.
###
# The paper "Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain" accepted at ECCV 2022.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-


import numpy as np
import torch
import sys
import random
import os
import logging

from sourcegen.argparser import make_parser
import sourcegen.utils as utils
import sourcegen.trainers as trainers


def main():
    # Setup seeds
    if args.training_seed is None:
        args.training_seed = np.random.randint(1, 10000)
    np.random.seed(args.training_seed)
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed(args.training_seed)
    torch.backends.cudnn.deterministic = True

    # Initialize logging
    os.makedirs(args.result_dir, exist_ok=True)
    log_filename = os.path.join(args.result_dir, 'logging.txt')
    logging.basicConfig(handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Load params
    args.dataset_spec = utils.misc.load_json(args.dataset_spec)
    args.method_spec = utils.misc.load_json(args.method_spec)

    # Setup trainer and run the method
    method_trainer = getattr(trainers, args.method_spec['model_spec']['name'])
    trainer = method_trainer(args)
    trainer.run()


if __name__ == '__main__':
    # Argument parser
    parser = make_parser()
    args = parser.parse_args()
    args.start_command = ' '.join(sys.argv)

    # Run
    main()


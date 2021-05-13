#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xinfa.jiang
# @File    : congif.py

import yaml
import os
from pathlib import Path


def load_yaml_config(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config


class Config:
    # root_dir = Path(os.path.dirname(__file__)).parents[0]
    root_dir = Path(os.path.dirname(__file__))
    config = load_yaml_config(os.path.join(root_dir, 'config.yaml'))

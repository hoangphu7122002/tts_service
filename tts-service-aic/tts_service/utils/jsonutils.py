#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18
import json


def to_json(obj: object) -> str:
    return json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False)

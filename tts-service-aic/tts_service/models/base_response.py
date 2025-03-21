#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2020/01/18


class BaseResponse:
    def __init__(self, msg=None, status=None):
        self.msg: str = msg
        self.status: int = status

    @staticmethod
    def from_dict(d):
        instance = BaseResponse()
        instance.__dict__.update(d)
        return instance

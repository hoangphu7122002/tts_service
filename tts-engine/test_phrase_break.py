#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @AUTHOR : thangdc94
# @Date : 2019/12/02
import unittest

from phrase_break import PhraseBreak


class TestProcessAll(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.phrase_break = PhraseBreak()

    def test_break_phrase(self):
        text = "râu_tơ đưa tin, quyết định của Nhà Trắng là nhằm đáp trả hạn chót mà các nghị sĩ đảng Dân chủ đưa ra buộc ông trăm phải ra điều trần vì những cáo buộc liên quan đến hành xử không chuẩn mực trong các thỏa thuận với u_cờ_rai_na."
        result = self.phrase_break.break_phrase(text)
        print(result)


if __name__ == '__main__':
    unittest.main()

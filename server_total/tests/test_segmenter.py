from __future__ import annotations

import unittest

from server_total.segmenter import PunctuationSegmenter


class SegmenterTests(unittest.TestCase):
    def test_sentence_endings_always_split_immediately(self):
        segmenter = PunctuationSegmenter()
        units = segmenter.feed("你好。可以吗？当然！继续")
        self.assertEqual(
            [(unit.text, unit.delimiter) for unit in units],
            [("你好。", "。"), ("可以吗？", "？"), ("当然！", "！")],
        )

    def test_short_comma_clauses_are_coalesced(self):
        segmenter = PunctuationSegmenter()
        units = segmenter.feed("处理数据、识别图像、理解语言等，帮助解决复杂问题。后续")
        self.assertEqual(
            [(unit.text, unit.delimiter) for unit in units],
            [
                ("处理数据、识别图像、理解语言等，", "，"),
                ("帮助解决复杂问题。", "。"),
            ],
        )

    def test_can_coalesce_short_sentences_for_streaming_tts(self):
        segmenter = PunctuationSegmenter(
            first_unit_min_chars=8,
            target_unit_chars=5,
            coalesce_hard_delimiters=True,
        )
        units = segmenter.feed("你好。可以吗？当然可以！下一段继续。尾部")
        self.assertEqual(
            [unit.text for unit in units],
            ["你好。可以吗？当然可以！", "下一段继续。"],
        )

    def test_first_unit_can_split_earlier_than_later_units(self):
        segmenter = PunctuationSegmenter(first_unit_min_chars=12, target_unit_chars=20)
        units = segmenter.feed(
            "人工智能能够快速处理数据，后续内容还比较短，需要继续累计，"
            "直到达到目标长度以后再进行切分，最后一句。尾部"
        )
        self.assertEqual(units[0].text, "人工智能能够快速处理数据，")
        self.assertEqual(units[0].delimiter, "，")
        self.assertGreaterEqual(len(units[1].text.rstrip("，")), 20)
        self.assertEqual(units[-1].text, "最后一句。")

    def test_groups_consecutive_marks_and_closing_quote(self):
        segmenter = PunctuationSegmenter()
        self.assertEqual(segmenter.feed("真的吗？！"), [])
        units = segmenter.feed("”下一句。后续")
        self.assertEqual(units[0].text, "真的吗？！”")
        self.assertEqual(units[0].delimiter, "？！”")
        self.assertEqual(units[1].text, "下一句。")

    def test_protects_numbers_versions_times_and_urls(self):
        segmenter = PunctuationSegmenter()
        units = segmenter.feed(
            "版本1.5在10:30发布，详情访问https://example.com，价格是3.14元。后续"
        )
        self.assertEqual(
            [unit.text for unit in units],
            ["版本1.5在10:30发布，详情访问https://example.com，", "价格是3.14元。"],
        )

    def test_flushes_short_final_clause_without_losing_delimiter(self):
        segmenter = PunctuationSegmenter()
        self.assertEqual(segmenter.feed("你好，"), [])
        units = segmenter.finish()
        self.assertEqual([(unit.text, unit.delimiter) for unit in units], [("你好，", "，")])

    def test_flushes_text_without_final_punctuation(self):
        segmenter = PunctuationSegmenter()
        self.assertEqual(segmenter.feed("没有结束符"), [])
        units = segmenter.finish()
        self.assertEqual([(unit.text, unit.delimiter) for unit in units], [("没有结束符", "")])


if __name__ == "__main__":
    unittest.main()

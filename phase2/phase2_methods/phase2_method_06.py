"""Phase-2 experiment 06: knowledge_distillation on Phase-1 base 03.

AI-assisted reference code; disclose its use under course rules.
"""
from phase2_method_common import EntityTagger, collate, predict


METHOD = 'knowledge_distillation'
BASE_ID = '03'


def build_task_model(base_model, tokenizer, labels, config, *, teacher_model=None,
                     teacher_tokenizer=None):
    return EntityTagger(base_model, BASE_ID, labels, METHOD, config, teacher=teacher_model)


def collate_batch(rows, tokenizer, labels, config, *, teacher_tokenizer=None):
    return collate(rows, tokenizer, labels, config,
                   teacher_tokenizer=teacher_tokenizer if METHOD == 'knowledge_distillation' else None)


def predict_spans(model, text, tokenizer, labels, config):
    return predict(model, text, tokenizer, labels, config)

import json

from test_experiment_contracts import sample
from test_prompting_runtime import Runner, config

from llm_grading.data.preprocess import build_task3_input
from llm_grading.prompting.pipeline import ModelPipeline


def test_task3_evidence_and_diagnosis_visible():
    row = sample()
    row.update(compile_log='compiler evidence', test_report='test evidence', diagnosis='confirmed bug')
    view = build_task3_input(row)
    assert view['compile_log'] == 'compiler evidence'
    assert view['test_report'] == 'test evidence'
    assert view['diagnosis'] == 'confirmed bug'
    assert 'feedback' not in view


def test_two_stage_public_without_labels(tmp_path):
    cfg = config(tmp_path, 'task3')
    cfg['prompt']['version'] = 'v004'
    cfg['task3'] = {'two_stage': True}
    row = sample()
    row.update(error_labels_missing=True, feedback_level=2)
    runner = Runner([json.dumps({'error_labels': ['Lỗi logic'], 'diagnosis': 'Điều kiện sai.'}),
                     'Trong hàm xử lý, điều kiện dừng khiến kết quả bị sai.'])
    result = ModelPipeline(cfg, runner).predict(row)
    assert result['error_labels'] == ['Lỗi logic']
    assert result['compliance']['pass']
    logs = [json.loads(line) for line in (tmp_path/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    assert logs[0]['stage'] == 'diagnosis'


def test_two_stage_rejects_invalid_taxonomy(tmp_path):
    import pytest
    cfg = config(tmp_path, 'task3')
    cfg['task3'] = {'two_stage': True}
    raw = json.dumps({'error_labels': ['invalid'], 'diagnosis': 'bug'})
    with pytest.raises(ValueError, match='Invalid diagnosis'):
        ModelPipeline(cfg, Runner([raw, raw])).predict(sample())

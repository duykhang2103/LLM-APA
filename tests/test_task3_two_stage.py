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


def test_compact_feedback_keeps_diagnosis_without_repeating_code(tmp_path):
    from llm_grading.prompting.formatter import render_task_prompt

    cfg = config(tmp_path, 'task3')
    cfg['prompt']['version'] = 'v006'
    cfg['task3'] = {'compact_feedback_input': True}
    row = sample()
    row.update(code='UNIQUE_STUDENT_CODE', diagnosis='Grounded diagnosis')
    prompt = render_task_prompt(row, 'task3', cfg, [
        {'input': {'feedback_level': 2, 'code': 'EXAMPLE_CODE'}, 'target': 'Teacher comment'}
    ])
    assert 'Grounded diagnosis' in prompt and 'Teacher comment' in prompt
    assert 'UNIQUE_STUDENT_CODE' not in prompt and 'EXAMPLE_CODE' not in prompt


def test_compact_diagnosis_sees_full_code(tmp_path):
    cfg = config(tmp_path, 'task3')
    cfg['task3'] = {'two_stage': True, 'compact_diagnosis_prompt': True,
                    'compact_feedback_input': True}
    cfg['prompt']['version'] = 'v006'
    prompts = []

    class CapturingRunner(Runner):
        def generate(self, prompt):
            prompts.append(prompt)
            return super().generate(prompt)

    row = sample()
    row.update(code='UNIQUE_STUDENT_CODE', feedback_level=2)
    runner = CapturingRunner([json.dumps({'error_labels': [], 'diagnosis': 'No confirmed defect'}),
                              'Bài làm được trình bày rõ ràng.'])
    ModelPipeline(cfg, runner).predict(row)
    assert 'UNIQUE_STUDENT_CODE' in prompts[0]
    assert 'UNIQUE_STUDENT_CODE' not in prompts[1]


def test_compact_two_stage_rag_initializes_with_train_without_diagnoses(tmp_path):
    from llm_grading.retrieval.retriever import Retriever

    cfg = config(tmp_path, 'task3')
    cfg['method'] = 'rag'
    cfg['prompt']['version'] = 'v006'
    cfg['task3'] = {'two_stage': True, 'compact_feedback_input': True,
                    'compact_diagnosis_prompt': True}
    cfg['retrieval'] = {'index': 'bm25', 'key': 'code', 'k': 1,
                        'exclude_duplicates': False}
    rows = [sample('train', 'int teacher_example() { return 1; }')]
    assert 'diagnosis' not in rows[0]
    retriever = Retriever(rows, 'task3', cfg)
    runner = Runner([json.dumps({'error_labels': [], 'diagnosis': 'No confirmed defect'}),
                     'Bài làm được trình bày rõ ràng.'])
    row = sample('query', 'int student_code() { return 2; }')
    row.update(error_labels_missing=True)
    result = ModelPipeline(cfg, runner, retriever).predict(row)
    assert result['compliance']['pass']
    logs = [json.loads(line) for line in (tmp_path/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    assert logs[-1]['retrieval']['retrieved'][0]['sample_id'] == 'train'


def test_diagnosis_alias_is_audited(tmp_path):
    cfg = config(tmp_path, 'task3')
    cfg['task3'] = {'two_stage': True}
    raw = json.dumps({'error_labels': ['Lỗi biên', 'Lỗi edge case'], 'diagnosis': 'A confirmed boundary defect.'})
    result = ModelPipeline(cfg, Runner([raw, 'Em hãy kiểm tra lại bài làm.'])).predict(sample())
    assert result['error_labels'] == ['Lỗi edge case']
    first = json.loads((tmp_path/'responses.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert first['taxonomy_normalization']['original'] == ['Lỗi biên', 'Lỗi edge case']


def test_diagnosis_failure_is_marked_without_praising_submission(tmp_path):
    cfg = config(tmp_path, 'task3')
    cfg['task3'] = {'two_stage': True, 'fallback_on_failure': True}
    raw = '{"error_labels": [], "diagnosis": "unfinished'
    result = ModelPipeline(cfg, Runner([raw, raw])).predict(sample())
    assert result['fallback'] and result['diagnosis_unavailable']
    assert 'đã đáp ứng' not in result['feedback']
    assert result['compliance']['pass']


def test_diagnosis_retry_includes_rejected_output(tmp_path):
    cfg = config(tmp_path, 'task3')
    cfg['task3'] = {'two_stage': True}
    prompts = []

    class Capture(Runner):
        def generate(self, prompt):
            prompts.append(prompt)
            return super().generate(prompt)

    raw = '{"error_labels": [], "diagnosis": "UNFINISHED'
    runner = Capture([raw, json.dumps({'error_labels': [], 'diagnosis': 'No defect confirmed.'}),
                      'Em hãy kiểm tra lại bài làm.'])
    ModelPipeline(cfg, runner).predict(sample())
    assert 'UNFINISHED' in prompts[1] and 'under 120 words' in prompts[1]

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/morai/run_bounded_diagnostics.py'
spec = importlib.util.spec_from_file_location('bounded_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def plan(tmp_path):
    script = tmp_path / 'child.py'
    script.write_text('pass\n')
    return dict(automatic_full_training=False, promote_models=False,
                preserved_sha256={str(script): queue.digest(script)},
                jobs=[dict(name='smoke', script=str(script), cwd=str(tmp_path),
                           args=['--smoke'], timeout_seconds=60)])


def test_valid_literal_plan(tmp_path):
    value = plan(tmp_path)
    assert queue.validate_plan(value, tmp_path) is value
    queue.check_preserved(value)


@pytest.mark.parametrize('field', ['automatic_full_training', 'promote_models'])
def test_no_automatic_expansion(tmp_path, field):
    value = plan(tmp_path)
    value[field] = True
    with pytest.raises(ValueError):
        queue.validate_plan(value, tmp_path)


def test_immutable_inputs(tmp_path):
    value = plan(tmp_path)
    Path(value['jobs'][0]['script']).write_text('changed\n')
    with pytest.raises(RuntimeError, match='changed'):
        queue.check_preserved(value)


@pytest.mark.parametrize('change', [dict(timeout_seconds=0), dict(timeout_seconds=20000),
                                    dict(args='python child.py'), dict(args=['--lock-fd', '1']),
                                    dict(name='../escape')])
def test_reject_unsafe_job(tmp_path, change):
    value = plan(tmp_path)
    value['jobs'][0].update(change)
    with pytest.raises(ValueError):
        queue.validate_plan(value, tmp_path)


def test_scope_escape(tmp_path):
    value = plan(tmp_path)
    value['jobs'][0]['cwd'] = str(tmp_path.parent)
    with pytest.raises(ValueError):
        queue.validate_plan(value, tmp_path)

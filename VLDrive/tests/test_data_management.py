"""Campaign registration admits sealed completed sources, never copies payloads."""
import json
from pathlib import Path
import pytest
from tools.manage_data import register, digest


def capture(tmp_path):
    source=tmp_path/'campaign'
    episode=source/'run/episode_000/episode.json'
    episode.parent.mkdir(parents=True)
    episode.write_text('{}')
    report=source/'validation/report.json'
    report.parent.mkdir()
    report.write_text(json.dumps(dict(schema='morai_sensor_record_campaign_validation_v1',
                                      valid=True, scenario_name='full_loop_normal', total_frames=10)))
    (report.parent/'COMPLETE.json').write_text(json.dumps(dict(status='passed', files={'report.json':digest(report)})))
    return source,report


def test_registration_preserves_source_and_rejects_duplicates(tmp_path):
    source,report=capture(tmp_path)
    before=digest(report)
    path=register(source,report,'normal','capture_01',root=tmp_path)
    record=json.loads(path.read_text())
    assert record['source']==str(source.resolve())
    assert len(record['episodes'])==1
    assert digest(report)==before
    with pytest.raises(ValueError,match='already registered'):
        register(source,report,'normal','capture_02',root=tmp_path)


def test_changed_report_or_wrong_category_rejected(tmp_path):
    source,report=capture(tmp_path)
    with pytest.raises(ValueError,match='Category'):
        register(source,report,'highway','capture',root=tmp_path)
    report.write_text(report.read_text()+' ')
    with pytest.raises(ValueError,match='sealed'):
        register(source,report,'normal','capture',root=tmp_path)


def test_collecting_rejected(tmp_path):
    source=tmp_path/'.collecting/campaign'
    source.mkdir(parents=True)
    with pytest.raises(ValueError,match='collecting'):
        register(source,source/'report.json','normal','capture',root=tmp_path)

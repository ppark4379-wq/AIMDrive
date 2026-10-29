"""Local RViz checks, separate from the static left-route integration."""
from AIM2026.tests.test_static_corridor import ROUTE, ego, planner
from AIM2026.tests.test_static_left_recordings import wall
from AIM2026.Visualization.snapshot import write_snapshot
from AIM2026.Visualization.rviz import display_state


def test_existing_rviz_snapshot_contains_the_generated_left_path(tmp_path):
    result = planner(wall(-62., -47.)).run(ego(), [], ROUTE)
    write_snapshot(tmp_path/'visualization.json', result, ego(), [], result.path)
    state = display_state(tmp_path, None)
    assert state['result'].reason == 'static_corridor_follow'
    assert min(p.x for p in state['selected_path']) < -63.
    assert any(c.selected_preview for c in state['result'].candidates)

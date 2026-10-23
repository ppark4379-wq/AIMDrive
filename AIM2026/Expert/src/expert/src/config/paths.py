"""Expert 파일 위치. 경로 좌표와 주행 파라미터는 여기서 변경하지 않는다."""
from pathlib import Path

EXPERT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIRECTORY = Path(__file__).resolve().parent
MAP_DIRECTORY = EXPERT_ROOT / "map_data"
RECORDED_PATHS_DIRECTORY = EXPERT_ROOT / "data" / "recorded_paths"
STATIC_OBSTACLE_PATHS = RECORDED_PATHS_DIRECTORY / "static_obstacle"
JAMMING_PATHS = RECORDED_PATHS_DIRECTORY / "jamming"
JAMMING_SECTION_CONFIG = CONFIG_DIRECTORY / "jamming_section.json"
RECORDINGS_DIRECTORY_NAME = "recordings"
PREPARED_DIRECTORY_NAME = "prepared"

import threading
import yaml

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AIM2026.UDP import UDPManager

class AIM2026:
    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else Path(__file__).with_name('config.yaml')
        self.config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))

        self.udp = UDPManager(self.config)
        self._stop = threading.Event()

    def run(self):
        self._stop.clear()
        try:
            self.udp.start()
            print("AIM2026 UDP 수신 시작: " + ', '.join(self.udp.receivers), flush=True)
            while not self._stop.wait(0.1):
                # samples = self.udp.snapshot()
                # 추론·제어 루프는 여기서 최신 샘플을 사용합니다.
                # self.udp.send_control(accel, brake, steer)
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.udp.close()

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    aim2026 = AIM2026()
    aim2026.run()
    

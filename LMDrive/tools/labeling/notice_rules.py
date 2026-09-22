"""CARLA 수집 기록에서 Notice의 '사건 프레임'을 자동 선택하는 규칙.

Scenario1~9는 CARLA 시나리오 타입이며 MORAI 시나리오 번호가 아니다.
입력 scenarios는 townXX_all_scenarios.json의 정의 데이터이고, frames는
measurements_full에 기록된 ego/actor 측정값이다. 시나리오 transform은
미리 정한 사건 트리거 위치이지 보행자의 실제 현재 위치나 충돌 지점이 아니다.
01~06은 트리거와 ego 기록의 거리만 검사하므로 사건 발생/가시성을 입증하지 않는다.
결과 frame_id는 frames 배열의 인덱스이다. 원시 simulator tick과 혼동하지 않는다.
이 파일은 영상 클립이나 문장을 생성하지 않는다. 실행 파서가 종류/ID를 붙이고,
학습 loader가 navigation 클립 안의 사건을 골라 문장 템플릿을 삽입한다.
MORAI 연동 설계와 수집팀 계약: VLDrive/docs/NOTICE_MORAI_INTEGRATION.md.
"""

import random
import math

class Notice():
    def __init__(self) -> None:
        self.notice_type = []

    def add_result(self, results, frames, transform):
        """한 트리거에 대해 전체 기록에서 가장 가까운 ego 프레임 하나를 찾는다.

        XY 유클리드 거리의 최솟값이 엄격히 2m 미만일 때만 자동 라벨링한다.
        최초 2m 진입 시점이 아니라 최근접 시점이다. actor 종류/속도/영상은
        검사하지 않으며, 동일 규칙 호출 안의 같은 frame_id 결과만 중복 제거한다.
        전체 기록을 사용하는 오프라인 정렬이므로 실시간 사건 검출기로 쓰지 않는다.
        """
        frame_num = len(frames)
        distance_min = float('inf')
        for index in range(frame_num):
            frame = frames[index]
            distance = math.sqrt(pow(transform["transform"]["x"]-frame["x"], 2)+pow(transform["transform"]["y"]-frame["y"], 2))
            if distance < distance_min:
                distance_min = distance
                index_min = index

        if distance_min < 2:
            result = {}
            result["frame_id"] = index_min
            result["instruction_args"] = []
            if not result in results:
                results.append(result)

    def choose_scenario_configs(self, scenario, scenarios_event_configs):
        # notice_type과 일치하는 CARLA 사건 정의의 트리거 목록을 선택한다.
        for notice in self.notice_type:
            if scenario["scenario_type"] == notice:
                scenario_event = scenario["available_event_configurations"]
                scenarios_event_configs.append(scenario_event)

    def process(self, data, scenarios, town_str):
        # 정의 JSON: available_scenarios[0][TownXX] -> scenario_type,
        # available_event_configurations -> 각 항목의 transform.x/y.
        results = []
        frames = data["data"]
        if town_str == "10":
            town_str = "10HD"
        town_str = "Town" + town_str
        scenarios_event_configs = []
        scenarios_data = scenarios["available_scenarios"][0][town_str]
        for scenario in scenarios_data:
            self.choose_scenario_configs(scenario, scenarios_event_configs)

        for scenario_event_configs in scenarios_event_configs:
            for transform in scenario_event_configs:
                self.add_result(results, frames, transform)

        return results

class Notice01(Notice):
    """Notice-01 / 현재 ID 50: 전방 보행자 주의. Scenario3 위치 기반 대리 라벨."""
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario3"]

class Notice02(Notice):
    """Notice-02 / 현재 ID 51: 전방 자전거 주의. Scenario4 위치 기반 대리 라벨."""
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario4"]

class Notice03(Notice):
    """Notice-03 / 현재 ID 52: 전방 차량 급정거 주의.

    Scenario2/5의 트리거만 사용한다. 앞 차량 속도 이력에서 감속도를 검출하지 않는다.
    """
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario2","Scenario5"]

class Notice04(Notice):
    """Notice-04 / 현재 ID 53: 좌측 신호위반 차량 주의.

    Scenario8/9로 매핑한다. 실제 차량의 신호위반/좌측 위치를 여기서 검증하지 않는다.
    """
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario8","Scenario9"]

class Notice05(Notice):
    """Notice-05 / 현재 ID 54: 전방 신호위반 차량 주의. Scenario7 트리거 기반."""
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario7"]

class Notice06(Notice):
    """Notice-06 / 현재 ID 55: 불량 노면 주의. Scenario1 트리거 기반."""
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario1"]

class Notice07():
    """Notice-07 / 현재 ID 56: 터널 주의.

    Town05의 고정 좌표 6곳에서만 동작한다. MORAI 터널 좌표로 해석하면 안 된다.
    15번 프레임부터 검사하여 거리 < 2m인 첫 프레임을 기록한다.
    검출 후 내부 +128과 루프 +1로 통상 129프레임 뒤부터 재검사한다.
    이는 중복 억제 간격이지 128프레임짜리 클립 생성이 아니다.
    """
    def __init__(self):
        self.scenario_loc = [[-211.35,-93.20],[-263.20,-69.99],[-211.31,-2.16],[-260.53,5.01],[-209.38,85.59],[-260.65,91.40]]

    def add_result(self,results,index,frames):
        frame = frames[index]
        for loc in self.scenario_loc:
            if math.sqrt(pow(loc[0]-frame["x"], 2)+pow(loc[1]-frame["y"], 2)) < 2:
                result = {}
                result["frame_id"] = index
                result["instruction_args"] = []
                results.append(result)
                index = min(index+128, len(frames)-1)
                return index
        return index

    def process(self, data, scenarios, town_str):
        results = []
        if town_str != "05":
            return results

        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1

        return results

class Notice08():
    """Notice-08-R/G/Y / 현재 ID 57/58/59: 전방 빨강/초록/노랑 신호 주의.

    ego에 연결된 affected_light_id의 actor가 있어야 하며 loc과 sta를 읽는다.
    CARLA 상태 인덱스 0/1/2는 빨강/노랑/초록이다. MORAI raw bitmask를
    그대로 이 배열의 인덱스로 넣으면 안 된다. 신호 전환 검출이 아니라 상태 검사다.
    색별 규칙은 독립 실행되며 15번부터 시작하고 검출 후 통상 129프레임 건너뛴다.
    """
    def __init__(self, light_status):
        self.light_status = light_status
        self.light_status_mapping = ['red','yellow','green']
        self.light_dis_town_mapping = {"01": 25,"02": 25,"03": 45,"04": 25,"05": 45,"06": 25,"07": 25,"10": 45}

    def add_result(self, results, index, frames, town_str):
        frame = frames[index]
        affected_light_id = frame["affected_light_id"]
        if affected_light_id == -1:
            return index
        if not str(affected_light_id) in frame["actors_data"].keys():
            return index
        light_loc = frame["actors_data"][str(affected_light_id)]["loc"]
        light_dis = math.ceil(math.sqrt(pow(light_loc[0]-frame["x"], 2)+pow(light_loc[1]-frame["y"], 2)))
        # 올림한 XY 거리 <= Town별 기본 거리 + U[0,10)일 때만 후보로 삼는다.
        # 기본값은 Town03/05/10에서 45m, 나머지 지원 Town에서 25m이다.
        # RNG seed를 이 모듈에서 고정하지 않으므로 실행마다 결과가 달라질 수 있다.
        if light_dis > self.light_dis_town_mapping[town_str] + random.random()*10:
            return index
        light_state = frame["actors_data"][str(affected_light_id)]["sta"]
        if self.light_status_mapping[light_state] == self.light_status:
            result = {}
            result["frame_id"] = index
            result["instruction_args"] = []
            results.append(result)
            index = min(index+128, len(frames)-1)
            return index
        return index

    def process(self, data, scenarios, town_str):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num:
            index = self.add_result(results, index, frames, town_str)
            index = index +1
        return results


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""합류 충돌구역 점유구간(occupancy interval) 기반 우선권 판단.

진입시점만 비교하면 "내가 먼저 들어갔는데 뒤에서 박히는" 상황을 막을 수
없다. Ego 가 구역을 **완전히 비우는 시점**과 상대차가 구역에 **들어오는
시점**을 비교해, 압도적으로 먼저 빠져나갈 수 있을 때만 GO 하고 조금이라도
겹치면 YIELD 한다.

    Ego     |<-- t_in_ego --->|===== 점유 =====|<- t_out_ego
    상대차                                       |<- t_in_other
                                              \___/
                                          이 여유(clear gap)가
                                          기준 이상이어야 GO

구역 안에 이미 들어간 뒤에는 절대 감속하지 않는다. 충돌구역 안에서 브레이크를
밟으면 상대차에게 측면·후면을 내주게 되므로, 진입 전 판단을 보수적으로 하고
진입 후에는 빠르게 비워주는 것이 안전하다.
"""

import math


def travel_time(distance, speed, accel, max_speed):
    """등가속 후 정속 프로파일로 distance 를 지나는 데 걸리는 시간.

    정지 직전의 Ego 에 현재 속도만으로 ETA 를 매기면 "느려서 못 끼어들고,
    못 끼어들어서 계속 느린" 합류 교착(deadlock)이 생긴다. Ego 는 실제로
    가속할 수 있으므로 가속을 반영해야 판단이 현실과 맞는다.
    """
    distance = float(distance)
    if distance <= 1e-6:
        return 0.0
    if math.isinf(distance):
        return float("inf")
    speed = max(0.0, float(speed))
    max_speed = max(1e-3, float(max_speed))
    accel = max(0.0, float(accel))
    if accel <= 1e-6:
        return distance / speed if speed > 1e-3 else float("inf")
    if speed >= max_speed:
        return distance / speed
    # 최고속도까지 가속하는 데 필요한 거리
    accel_distance = (max_speed * max_speed - speed * speed) / (2.0 * accel)
    if distance <= accel_distance:
        return (
            math.sqrt(speed * speed + 2.0 * accel * distance) - speed
        ) / accel
    accel_phase_time = (max_speed - speed) / accel
    return accel_phase_time + (distance - accel_distance) / max_speed


class MergeZoneConfig(object):
    """한 합류 충돌구역의 기하 정보와 판단 파라미터."""

    def __init__(
        self,
        name,
        center,
        radius,
        successors,
        ego_watch_radius,
        max_speed,
        stop_margin,
        comfort_decel,
        follow_time_gap,
        clear_pass_distance,
        go_clear_time_margin,
        go_min_lead_distance,
        go_rear_time_gap,
        go_assumed_accel,
        go_cruise_speed,
        hold_release_ratio,
        decision_horizon,
        dropout_hold_time,
        yield_max_decel,
        commit_min_speed,
    ):
        self.name = name
        self.center = center
        self.radius = float(radius)
        self.successors = frozenset(successors)
        self.ego_watch_radius = float(ego_watch_radius)
        self.max_speed = float(max_speed)
        self.stop_margin = float(stop_margin)
        self.comfort_decel = float(comfort_decel)
        self.follow_time_gap = float(follow_time_gap)
        self.clear_pass_distance = float(clear_pass_distance)
        self.go_clear_time_margin = float(go_clear_time_margin)
        self.go_min_lead_distance = float(go_min_lead_distance)
        self.go_rear_time_gap = float(go_rear_time_gap)
        self.go_assumed_accel = float(go_assumed_accel)
        self.go_cruise_speed = float(go_cruise_speed)
        self.hold_release_ratio = float(hold_release_ratio)
        self.decision_horizon = float(decision_horizon)
        self.dropout_hold_time = float(dropout_hold_time)
        self.yield_max_decel = float(yield_max_decel)
        self.commit_min_speed = float(commit_min_speed)


class MergeOccupancy(object):
    """한 차량이 충돌구역을 점유하는 시간 구간."""

    __slots__ = (
        "key",
        "speed",
        "entry_distance",
        "exit_distance",
        "entry_time",
        "exit_time",
    )

    def __init__(
        self,
        key,
        speed,
        entry_distance,
        exit_distance,
        entry_time,
        exit_time,
    ):
        self.key = key
        self.speed = float(speed)
        self.entry_distance = float(entry_distance)
        self.exit_distance = float(exit_distance)
        self.entry_time = float(entry_time)
        self.exit_time = float(exit_time)

    @property
    def inside(self):
        return self.entry_distance <= 1e-6

    def __repr__(self):
        def show(value):
            return "inf" if math.isinf(value) else "{:.2f}".format(value)

        return "{}[in={}s out={}s v={:.1f}]".format(
            self.key, show(self.entry_time), show(self.exit_time), self.speed
        )


class MergeZoneState(object):
    """합류 구역 하나에 대한 우선권 상태기.

    GO/YIELD 는 기존 디버그·시각화와의 호환을 위해 "EGO"/"OTHER" 문자열을
    유지한다.
    """

    GO = "EGO"
    YIELD = "OTHER"

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.mode = None
        self.committed = False
        self.blocking_key = None
        self._dropout_hold = 0.0
        self._dropout_target_time = float("inf")

    def ego_travel_time(self, distance, ego_speed):
        """Ego 는 가속 가능성을 반영해 구역 통과시간을 예측한다."""
        return travel_time(
            distance,
            ego_speed,
            self.config.go_assumed_accel,
            max(self.config.go_cruise_speed, ego_speed),
        )

    def update(self, ego, opponents, ego_speed, cycle_time):
        """(mode, speed_limit, debug_lines) 를 돌려준다."""
        config = self.config
        debug = []

        # 1) 이미 충돌구역 안이면 무조건 비워준다. 구역 안에서의 감속은
        #    측면·후면 추돌을 자초하므로 어떤 경우에도 하지 않는다.
        if ego.inside:
            if self.mode != self.GO:
                debug.append(
                    "{}: ego inside zone -> commit GO (clear it, never brake)".format(
                        config.name
                    )
                )
            self.mode = self.GO
            self.committed = True
            self.blocking_key = None
            self._dropout_hold = 0.0
            self._dropout_target_time = float("inf")
            return self.GO, float("inf"), debug

        if self.committed:
            # 구역을 통과해 나왔으니 다음 판단을 새로 시작한다.
            self.committed = False
            self.mode = None

        # 2) 양보를 하려면 구역 경계 앞에 멈출 수 있어야 한다. 이미 멈출 수
        #    없는 거리라면 양보는 선택지가 아니다. 어정쩡하게 감속해 구역
        #    안에서 멈추는 것이 가장 위험하므로 통과를 확정한다.
        #    (감속하면 정지거리가 함께 줄어들어 이 판정은 스스로 뒤집히지
        #     않는다. 즉 한 번 양보를 시작하면 양보 가능성이 유지된다.)
        stopping_distance = ego_speed * ego_speed / (
            2.0 * max(0.1, config.yield_max_decel)
        )
        if (
            ego_speed > config.commit_min_speed
            and ego.entry_distance - config.stop_margin < stopping_distance
        ):
            if self.mode != self.GO:
                debug.append(
                    "{}: point of no return entry_d={:.1f}m < stop_d={:.1f}m"
                    " -> commit GO".format(
                        config.name, ego.entry_distance, stopping_distance
                    )
                )
            self.mode = self.GO
            self.committed = True
            self.blocking_key = None
            return self.GO, float("inf"), debug

        # 3) 판단 지평 안에 실제로 구역에 들어오는 상대차만 고려한다.
        #    이 필터가 없으면 멀리 있는 차 때문에 상시 감속하게 된다.
        candidates = [
            other
            for other in opponents
            if other.entry_time <= config.decision_horizon + 1e-9
        ]

        # GO 를 유지 중일 때는 기준을 완화해 잡음으로 판단이 흔들리지 않게 한다.
        required_margin = config.go_clear_time_margin
        if self.mode == self.GO:
            required_margin *= config.hold_release_ratio

        blocking = []
        for other in candidates:
            # Ego 가 구역을 완전히 비우고 나서 상대차가 들어오기까지의 여유
            clear_gap = other.entry_time - ego.exit_time
            if clear_gap < required_margin:
                blocking.append((other, clear_gap, "overlap"))
                continue
            # 시간 여유가 있어도 합류 직후 후방 간격이 모자라면 추돌당한다.
            lead_distance = clear_gap * max(0.0, ego_speed)
            required_lead = (
                config.go_min_lead_distance
                + config.go_rear_time_gap
                * max(0.0, other.speed - ego_speed)
            )
            if lead_distance < required_lead:
                blocking.append((other, clear_gap, "rear_gap"))

        if blocking:
            self._dropout_hold = config.dropout_hold_time
            return self._yield(ego, blocking, debug)

        if self.mode == self.YIELD and self._dropout_hold > 1e-9:
            # 막고 있던 차량이 한두 프레임 사라졌을 뿐일 수 있다. 직전
            # 목표시간을 이어서 쓰며 잠시 양보를 유지한다.
            self._dropout_hold = max(0.0, self._dropout_hold - cycle_time)
            if not math.isinf(self._dropout_target_time):
                self._dropout_target_time = max(
                    0.0, self._dropout_target_time - cycle_time
                )
            debug.append(
                "{}: blocking vehicle dropped out, holding YIELD for {:.2f}s".format(
                    config.name, self._dropout_hold
                )
            )
            return (
                self.YIELD,
                self._yield_speed_limit(ego, self._dropout_target_time),
                debug,
            )

        return self._go(ego, candidates, debug)

    def _go(self, ego, candidates, debug):
        config = self.config
        if self.mode != self.GO:
            debug.append(
                "{}: GO ego_out={} vs others={} (clear margin met)".format(
                    config.name,
                    _show(ego.exit_time),
                    ",".join(repr(other) for other in candidates) or "none",
                )
            )
        self.mode = self.GO
        self.blocking_key = None
        return self.GO, float("inf"), debug

    def _yield_speed_limit(self, ego, target_time):
        """구역 경계 앞에서 멈출 수 있는 목표속도.

        정지 곡선은 멀리서는 크고 경계에 가까워지면 0 으로 수렴하므로
        먼 거리에서 목표속도가 갑자기 0 으로 떨어지는 급제동이 없다.
        """
        config = self.config
        stopping_limit = math.sqrt(
            2.0
            * config.comfort_decel
            * max(0.0, ego.entry_distance - config.stop_margin)
        )
        speed_limit = min(config.max_speed, stopping_limit)
        if not math.isinf(target_time):
            # 상대차가 비우기 전에 너무 일찍 도착하지 않도록 페이싱 상한.
            speed_limit = min(
                speed_limit, ego.entry_distance / max(0.1, target_time)
            )
        return max(0.0, speed_limit)

    def _yield(self, ego, blocking, debug):
        config = self.config
        worst = min(blocking, key=lambda item: item[1])
        self.blocking_key = worst[0].key
        self.mode = self.YIELD

        # 막고 있는 차량 전부가 구역을 비운 뒤에 도착하도록 목표시간을 잡는다.
        target_time = float("-inf")
        for other, _, _ in blocking:
            target_time = max(target_time, other.exit_time)
        if math.isinf(target_time) and target_time < 0:
            target_time = float("inf")
        if not math.isinf(target_time):
            target_time += config.follow_time_gap
        self._dropout_target_time = target_time

        speed_limit = self._yield_speed_limit(ego, target_time)

        debug.append(
            "{}: YIELD by={} reason={} clear_gap={} ego_in={}s ego_out={}s "
            "other_in={}s other_out={}s entry_d={:.1f}m -> limit={:.2f}m/s".format(
                config.name,
                worst[0].key,
                worst[2],
                _show(worst[1]),
                _show(ego.entry_time),
                _show(ego.exit_time),
                _show(worst[0].entry_time),
                _show(worst[0].exit_time),
                ego.entry_distance,
                speed_limit,
            )
        )
        return self.YIELD, speed_limit, debug


def _show(value):
    return "inf" if math.isinf(value) else "{:.2f}".format(value)

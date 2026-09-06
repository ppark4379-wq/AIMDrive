#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import io
import csv
from ..localization.point import Point
from ..path.mapping import NodeTrafficLightStoplineMapper


class Config(object):
    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)

            config_path = os.environ.get(
                'GT_BEV_CONFIG_PATH',
                os.path.join(os.path.dirname(__file__), 'config.json'),
            )
            with io.open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                cls._instance.__dict__ = config
                cls._instance._config_path = os.path.abspath(config_path)

            cls._instance._set_map_data()
        return cls._instance

    def __getitem__(self, key):
        return getattr(self, key)

    def _set_map_data(self):
        workspace_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../../',
        ))
        default_runtime_path = os.path.join(
            workspace_root, '.runtime', 'path.csv'
        )
        legacy_runtime_path = os.path.join(
            workspace_root, '.runtime', 'scenario_runner', 'path.csv'
        )
        if (
            not os.path.exists(default_runtime_path)
            and os.path.exists(legacy_runtime_path)
        ):
            default_runtime_path = legacy_runtime_path
        path_csv = os.environ.get(
            'GT_BEV_PATH_CSV',
            default_runtime_path,
        )
        with io.open(path_csv, 'r', encoding='utf-8', newline='') as f:
            self["map"]["path"] = [
                Point(float(row["x"]), float(row["y"]))
                for row in csv.DictReader(f)
            ]

        gt_bev_root = workspace_root
        bundled_map_data_dir = os.path.join(
            gt_bev_root,
            'map_data/R_KR_PG_KATRI_2025',
        )
        default_map_data_dir = os.path.normpath(os.path.join(
            gt_bev_root,
            self["map"].get("data_dir", "map_data/R_KR_PG_KATRI_2025"),
        ))
        map_data_dir = os.environ.get('GT_BEV_MAP_DIR', default_map_data_dir)
        if not os.path.isabs(map_data_dir):
            map_data_dir = os.path.abspath(map_data_dir)

        stopline_mapper = NodeTrafficLightStoplineMapper(map_data_dir)
        if len(stopline_mapper) == 0 and os.path.abspath(map_data_dir) != os.path.abspath(bundled_map_data_dir):
            print(
                "[TrafficSignal] selected map has no direct stop-node mapping; "
                "using bundled MGeo v3 data: {}".format(bundled_map_data_dir)
            )
            map_data_dir = bundled_map_data_dir
            stopline_mapper = NodeTrafficLightStoplineMapper(map_data_dir)
        if len(stopline_mapper) == 0:
            raise ValueError(
                "No traffic_light_id stop nodes found in: {}".format(map_data_dir)
            )

        self._resolve_speed_limit_zones(map_data_dir)
        self["map"]["data_dir"] = map_data_dir
        self["map"]["stopline_mapper"] = stopline_mapper
        # Traffic lights are no longer injected into ACC as averaged static
        # objects.  Red-light control selects the lane-centre stop node on the
        # active global path instead.
        self["map"]["traffic_light_list"] = []
        print(
            "[TrafficSignal] config: {}".format(self._config_path)
        )
        print(
            "[TrafficSignal] map data: {}, signal_ids={}, stop_nodes={}".format(
                map_data_dir,
                len(stopline_mapper),
                stopline_mapper.stop_node_count,
            )
        )

    def _resolve_speed_limit_zones(self, map_data_dir):
        velocity_config = self["planning"].get("velocity_profile", {})
        zones = velocity_config.get("speed_limit_zones", [])
        if not zones:
            return

        link_path = os.path.join(map_data_dir, "link_set.json")
        with io.open(link_path, 'r', encoding='utf-8') as f:
            link_by_id = {
                str(link.get("idx")): link
                for link in json.load(f)
                if link.get("idx") is not None
            }

        for zone in zones:
            start_id = str(zone["start_link_id"])
            end_id = str(zone["end_link_id"])
            if start_id not in link_by_id or end_id not in link_by_id:
                raise KeyError(
                    "Speed-limit zone link not found: {} -> {}".format(
                        start_id, end_id
                    )
                )

            start_points = link_by_id[start_id].get("points", [])
            end_points = link_by_id[end_id].get("points", [])
            if not start_points or not end_points:
                raise ValueError(
                    "Speed-limit zone link has no points: {} -> {}".format(
                        start_id, end_id
                    )
                )

            zone["start_point"] = start_points[0]
            zone["end_point"] = end_points[-1]
            print(
                "[VelocityProfile] zone {}: {} start -> {} end, limit={}km/h".format(
                    zone.get("name", "unnamed"),
                    start_id,
                    end_id,
                    float(zone["max_velocity"]),
                )
            )
    def update_config(self, file_name):
        with io.open(file_name, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.__dict__.update(config)

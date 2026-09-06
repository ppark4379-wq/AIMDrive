#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..localization.point import Point


class ObjectInfo:
    def __init__(
        self,
        x,
        y,
        velocity,
        object_type,
        name="",
        is_static=False,
        size_x=0.0,
        size_y=0.0,
        heading_deg=0.0,
    ):
        """Simulator 에서 얻어지는 object 정보를 담는 data class"""
        self.position = Point(x, y)
        self.velocity = velocity
        self.type = object_type  # 0: person / 1, 2: vehicle / 3: traffic light
        self.name = name
        # MORAI uses separate npc_list/obstacle_list/pedestrian_list arrays.
        # Preserve the source array so planning can activate on static objects only.
        self.is_static = bool(is_static)
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.heading_deg = float(heading_deg)

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

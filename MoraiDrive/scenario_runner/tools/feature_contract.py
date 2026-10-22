def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_feature_contract(package_root, scenario_cfg):
    expected_data_files = {
        "data/R_KR_PR_K-city_2025/link_set.json",
        "data/R_KR_PR_K-city_2025/node_set.json",
        "data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt",
    }
    actual_data_files = {
        path.relative_to(package_root).as_posix()
        for path in (package_root / "data").rglob("*")
        if path.is_file()
    }
    require(
        actual_data_files == expected_data_files,
        "Separated data boundary is not exact: "
        f"expected={sorted(expected_data_files)}, "
        f"actual={sorted(actual_data_files)}",
    )

    profile_weather = [
        str(profile["weather"]).upper()
        for profile in scenario_cfg.get("profiles", [])
    ]
    require(
        profile_weather == ["SUNNY", "FOGGY"],
        f"Weather alternation contract changed: {profile_weather}",
    )
    require(
        list(scenario_cfg.get("scenario_times", [])) == [11, 13, 15],
        "Scenario time choices must remain 11/13/15",
    )

    expected_static_models = {
        "CargoBox",
        "WoodBox",
        "RedBarrel",
        "YellowBarrel",
        "TrafficBarrel",
        "Lane_Separation_Bar",
        "PE_Drum",
        "PE_Firewall_Orange",
        "PE_Firewall_White",
        "Steel_Barricade",
        "Concrete_Barricade01",
        "Concrete_Barricade02",
        "Bollard01",
        "Bollard04",
        "Bicycle_city",
        "Bicycle_mountain",
        "Scooter_02",
        "Sportbike_01",
        "BabyStroller1",
        "BabyStroller2",
        "BabyStroller3",
        "Bike1",
        "Bike2",
        "SportBike",
        "ElectricScooter1",
        "ElectricScooter2",
        "obj_obstacle_dog",
        "NCAP_GVT",
        "Color_Cone",
        "Rubber_Cone",
        "concrete-barrier",
        "Bike3",
        "Scooter1",
        "Scooter2",
        "Rockslide",
        "Barbed_Wirefence",
    }
    static_models = set(scenario_cfg.get("static_obstacle_models", []))
    require(
        static_models == expected_static_models,
        "Native static-obstacle audit set changed: "
        f"missing={sorted(expected_static_models - static_models)}, "
        f"unexpected={sorted(static_models - expected_static_models)}",
    )

    require(
        bool(scenario_cfg.get("dynamic_obstacle_models")),
        "Dynamic-obstacle model pool is missing",
    )

    if scenario_cfg.get("mission_mode") == "full_loop_traffic_light":
        from ..scenarios.segment_modes.traffic_light import default_contract
        contract = scenario_cfg.get("_traffic_light_collection", {})
        require(scenario_cfg.get("traffic_light_wait_override", {}).get("enabled") is False,
                "좌회전 신호 수집에서는 기존 10초 뒤 직진 초록 override를 꺼야 합니다")
        expected = default_contract()
        require(all(contract.get(key) == value for key, value in expected.items()
                    if key != "source_sha256"), "좌회전 신호 수집 행동 계약이 다릅니다")
        pins = contract.get("source_sha256", {})
        require(all(pins.get(name) == sha for name, sha in expected["source_sha256"].items()),
                "좌회전 신호 수집 소스 버전이 다릅니다")
    else:
        traffic_override = scenario_cfg.get("traffic_light_wait_override", {})
        require(
            (traffic_override.get("enabled") is False
             if scenario_cfg.get("ve_full_route_signals") and scenario_cfg.get("ve_signal_head")
             else bool(traffic_override.get("enabled"))),
            "VE full-route signals require natural phases; other runs retain the wait override",
        )
        require(
            float(traffic_override.get("wait_wall_sec", -1.0)) == 10.0,
            "Traffic-light override must wait 10 host-monotonic seconds",
        )
        require(
            float(traffic_override.get("stopped_speed_mps", -1.0)) == 0.5
            and float(traffic_override.get("minimum_stopline_distance_m", -1.0)) == 0.0
            and float(traffic_override.get("maximum_stopline_distance_m", -1.0)) == 25.0,
            "Traffic-light stop and distance thresholds changed",
        )
        require(
            int(traffic_override.get("target_color", -1)) == 16
            and bool(traffic_override.get("impulse"))
            and bool(traffic_override.get("set_sibling")),
            "Traffic-light override command contract changed",
        )

    roundabout = scenario_cfg.get("roundabout_npc", {})
    require(
        int(roundabout.get("min_count", 0)) == 3
        and int(roundabout.get("max_count", 0)) == 5,
        "Roundabout NPC count range must remain 3..5",
    )
    require(
        float(roundabout.get("speed_mps", 0.0)) == 4.0,
        "Roundabout NPC speed must remain fixed at 4.0m/s",
    )
    require(
        int(roundabout.get("route_laps", 0)) == 100,
        "Roundabout NPC route must remain fixed at 100 laps",
    )
    require(
        list(roundabout.get("models", [])) == [
            "DEFAULT_SEDAN",
            "2016_Hyundai_Sonata",
            "2020_hyundai_sonata",
            "2015_Kia_K5",
            "2020_Kia_K5",
            "2017_Hyundai_Grandeur",
            "2014_Kia_K7",
            "2016_Hyundai_Genesis_DH",
            "2019_Toyota_Prius",
            "2021_Volkswagen_Golf_GTI",
            "2020_Kia_Stinger",
            "2016_Hyundai_Ioniq",
            "2023_Hyundai_Ioniq5",
            "2023_Kia_EV6",
            "2019_Hyundai_Kona(EV)",
            "2017_Kia_Niro(HEV)",
            "2020_Kia_Niro(EV)",
            "2016_Kia_Soul(EV)",
            "2020_Kia_Soul(EV)",
            "2021_Hyundai_G80",
            "2019_hyundai_genesis_g90",
            "2019_hyundai_genesis_g90_4WS",
            "DEFAULT_SUV",
            "2016_Hyundai_Santafe",
            "2017_Kia_Sorento",
            "2020_volvo_xc90",
            "2021_Hyundai_Genesis_GV80",
            "2021_Hyundai_GV70",
            "2019_Hyundai_Nexo",
            "DEFAULT_MPV",
            "DEFAULT_WAGON",
            "2019_kia_sedona",
            "2021_Kia_Carnival",
            "2020_chrysler_pacifica_hybrid",
            "2017_hyundai_solati_h350",
            "DR_KGTorres_EVX_2024",
        ],
        "Roundabout NPCs must use the approved model pool",
    )
    nonuniform = roundabout.get("nonuniform_position", {})
    require(
        float(nonuniform.get("min_spawn_gap_m", 0.0)) == 10.0
        and float(nonuniform.get("tight_gap_max_m", 0.0)) == 12.0
        and float(nonuniform.get("gap_weight_exponent", 0.0)) == 2.5,
        "Nonuniform roundabout spawn-gap contract changed",
    )
    speed_random = roundabout.get("speed_randomization", {})
    require(
        float(speed_random.get("min_speed_mps", 0.0)) == 2.5
        and float(speed_random.get("max_speed_mps", 0.0)) == 7.0
        and float(speed_random.get("slow_band_max_mps", 0.0)) == 3.2
        and float(speed_random.get("fast_band_min_mps", 0.0)) == 6.3
        and float(speed_random.get("interaction_gap_min_m", 0.0)) == 35.0
        and float(speed_random.get("interaction_gap_max_m", 0.0)) == 45.0
        and float(speed_random.get("live_log_period_s", 0.0)) == 0.5
        and float(speed_random.get("emergency_gap_m", 0.0)) == 6.0
        and float(speed_random.get("follow_gap_m", 0.0)) == 9.0
        and float(speed_random.get("resume_gap_m", 0.0)) == 12.0
        and float(speed_random.get("control_period_s", 0.0)) == 0.2,
        "Random-speed roundabout safety controller contract changed",
    )

    actor_lifecycle = scenario_cfg.get("actor_lifecycle", {})
    expected_actor_lifecycle = {
        "version": "nearby_actors_v3",
        "roundabout_lead_m": 100.0,
        "roundabout_retire_after_m": 30.0,
        "roundabout_stall_window_s": 45.0,
        "roundabout_stall_min_progress_m": 5.0,
        "roundabout_min_active_scenario_s": 20.0,
        "roundabout_mutation_quarantine_s": 1.0,
        "highway_stop_after_entry_m": 220.0,
        "highway_spawn_distance_m": 220.0,
    }
    require(
        actor_lifecycle == expected_actor_lifecycle,
        f"Actor lifecycle policy changed: {actor_lifecycle}",
    )

    highway = scenario_cfg.get("highway_npc", {})
    spawn_points = list(highway.get("spawn_points", []))
    require(
        [int(spec.get("id", -1)) for spec in spawn_points] == [10, 6, 8],
        "Highway SpawnPoints must exclude the roundabout-area sample point",
    )
    require(
        float(highway.get("background_warmup_distance_m", -1.0)) == 100.0,
        "Highway SpawnPoints must activate 100m before their activation point",
    )
    require(
        int(highway.get("collection_maximum_total_vehicles", -1)) == 12,
        "Highway total vehicle cap must remain 12",
    )
    require(
        all(not bool(spec.get("is_lane_change", True)) for spec in spawn_points),
        "Highway native SpawnPoints must keep lane changes disabled",
    )
    require(
        all(int(spec.get("maximum_spawn_vehicle", 0)) == 4 for spec in spawn_points),
        "Each highway SpawnPoint maximum vehicle count must remain 4",
    )
    require(
        all(bool(spec.get("pause")) for spec in spawn_points),
        "Highway SpawnPoints must be created paused",
    )
    require(
        all(
            float(spec.get("min_spawn_period_sec", -1.0)) == 5.0
            and float(spec.get("max_spawn_period_sec", -1.0)) == 5.0
            for spec in spawn_points
        ),
        "Highway SpawnPoint periods must remain fixed at 5 seconds",
    )
    require(
        float(highway.get("reference_safety_distance_m", 0.0)) == 5.0
        and float(highway.get("reference_ttc_sec", 0.0)) == 1.0,
        "Competition reference SafetyDist/TTC values changed",
    )
    require(
        not any(str(key).startswith("lead_") for key in highway),
        "Scripted highway lead settings must stay removed",
    )
    merge_static = scenario_cfg.get("highway_merge_static_vehicle", {})
    merge_static_models = [
        "DEFAULT_SEDAN",
        "2014_Kia_K7",
        "2015_Kia_K5",
        "2015_Toyota_Prius(HEV)",
        "2016_Hyundai_Genesis_DH",
        "2016_Hyundai_Ioniq",
        "2016_Hyundai_Sonata",
        "2017_Hyundai_Grandeur",
        "2019_Toyota_Prius",
        "2020_Kia_K5",
        "2020_Kia_Stinger",
        "2020_hyundai_sonata",
        "2021_Hyundai_G80",
    ]
    require(
        (bool(merge_static.get("enabled")) or (
            scenario_cfg.get("highway_merge_static_vehicle_explicitly_disabled") is True
            and merge_static.get("enabled") is False))
        and merge_static.get("models") == merge_static_models
        and merge_static.get("route_link") == "A2256W000411"
        and float(merge_static.get("distance_from_link_start_m", -1.0)) == 70.0,
        "Highway merge stationary sedan contract changed",
    )
    require(
        bool(scenario_cfg.get("tunnel_sudden_brake", {}).get("enabled")),
        "Tunnel sudden-brake event must be enabled",
    )
    require(
        int(scenario_cfg.get("route_setup_attempts", -1)) == 1,
        "Route setup must remain a single attempt during collection",
    )
    if scenario_cfg.get("mission_mode") == "full_loop_traffic_light":
        require(float(scenario_cfg.get("timeout_sec", 0)) == 180.0
                and float(scenario_cfg.get("off_route_timeout_sec", 0)) == 3.0
                and scenario_cfg.get("progress_watchdog", {}).get("enabled") is True
                and float(scenario_cfg.get("expert_max_velocity_kmh", 0)) == 30.0,
                "신호 수집 구간의 시간/이탈/속도 제한이 다릅니다")
    else:
        require(
            float(scenario_cfg.get("timeout_sec", -1.0)) == 0.0,
            "Scenario-internal overall timeout must remain disabled",
        )
        require(
            not bool(scenario_cfg.get("progress_watchdog", {}).get("enabled", True)),
            "Scenario progress watchdog must remain disabled",
        )
        require(
            float(scenario_cfg.get("off_route_timeout_sec", 0.0)) == 1000000000000.0,
            "Off-route timeout must remain effectively disabled",
        )
    require(
        int(scenario_cfg.get("after_static_start_path_index", -1)) >= 0,
        "After-static START index is missing",
    )
    require(
        int(scenario_cfg.get("highway_tunnel_test_start_path_index", -1)) >= 0,
        "Highway/tunnel test START index is missing",
    )
    if scenario_cfg.get("mission_mode") == "full_loop_traffic_light":
        require(scenario_cfg.get("auto_repeat") is False,
                "신호 파일럿은 한 회차 종료 후 자동 반복하지 않습니다")
    else:
        require(
            bool(scenario_cfg.get("auto_repeat")),
            "Automatic next-scenario execution must remain enabled",
        )

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

    static_models = set(scenario_cfg.get("static_obstacle_models", []))
    required_static_models = {
        "WoodBox",
        "Scooter_02",
        "Bicycle_city",
        "Steel_Barricade",
    }
    require(
        required_static_models <= static_models,
        "Compact static-obstacle families were not fully separated",
    )
    require(
        "RedBarrel" not in static_models,
        "RedBarrel must remain excluded from static-obstacle selection",
    )
    require(
        bool(scenario_cfg.get("dynamic_obstacle_models")),
        "Dynamic-obstacle model pool is missing",
    )

    traffic_override = scenario_cfg.get("traffic_light_wait_override", {})
    require(
        bool(traffic_override.get("enabled")),
        "Traffic-light wait override must remain enabled",
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
        int(roundabout.get("count", 0)) == 3,
        "Roundabout NPC count must remain fixed at 3",
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
        roundabout.get("model") == "DEFAULT_SEDAN",
        "Roundabout NPCs must remain passenger-car sized",
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
        bool(merge_static.get("enabled"))
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
    require(
        bool(scenario_cfg.get("auto_repeat")),
        "Automatic next-scenario execution must remain enabled",
    )

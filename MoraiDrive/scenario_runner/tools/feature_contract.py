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
        bool(scenario_cfg.get("dynamic_obstacle_models")),
        "Dynamic-obstacle model pool is missing",
    )

    roundabout = scenario_cfg.get("roundabout_npc", {})
    require(
        int(roundabout.get("count", 0)) == 6,
        "Roundabout NPC count must remain fixed at 6",
    )
    require(
        float(roundabout.get("speed_mps", 0.0)) == 4.0,
        "Roundabout NPC speed must remain fixed at 4.0m/s",
    )
    require(
        roundabout.get("model") == "DEFAULT_SEDAN",
        "Roundabout NPCs must remain passenger-car sized",
    )

    highway = scenario_cfg.get("highway_npc", {})
    spawn_points = list(highway.get("spawn_points", []))
    require(
        [int(spec.get("id", -1)) for spec in spawn_points] == [10, 6, 8],
        "Highway SpawnPoints must exclude the roundabout-area sample point",
    )
    require(
        all(not bool(spec.get("is_lane_change", True)) for spec in spawn_points),
        "Highway native SpawnPoints must keep lane changes disabled",
    )
    require(
        all(int(spec.get("maximum_spawn_vehicle", 0)) == 50 for spec in spawn_points),
        "Highway native SpawnPoint maximum vehicle count changed",
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
    require(
        bool(scenario_cfg.get("tunnel_sudden_brake", {}).get("enabled")),
        "Tunnel sudden-brake event must be enabled",
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

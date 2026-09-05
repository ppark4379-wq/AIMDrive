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
        "Tree01",
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
    require(
        int(highway.get("count_min", 0)) >= 1
        and int(highway.get("count_max", 0))
        >= int(highway.get("count_min", 0)),
        "Highway NPC count range is invalid",
    )
    require(
        float(highway.get("lead_gap_m", 0.0)) > 0.0,
        "Highway lead interaction settings are missing",
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

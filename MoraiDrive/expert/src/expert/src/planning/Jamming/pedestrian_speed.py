"""Existing validated-jamming-path pedestrian speed decision."""
from .pedestrian_safety import pedestrian_conflict
from ...config.parameters import highway as highway_parameters


def pedestrian_speed(self, static_valid, lattice_result, planning_path, vehicle_state, planned_velocity, planning_object_info_dic_list):
    person_check = None
    pedestrian_target_velocity = float('inf')
    if static_valid and lattice_result.debug_details.get('jamming'):
        # Replace the broad legacy person gate only on a validated tunnel
        # path. Include stationary people and actors outside the front gate.
        # Predict a restart, not the previous ACC stop command.
        person_conflicts = {
            id(entry['object_info']): pedestrian_conflict(
                planning_path, vehicle_state, entry['object_info'],
                min(max(0., planned_velocity), 30./3.6))
            for entry in planning_object_info_dic_list
            if entry['object_info'].type == 0
        }
        person_check = lambda person: person_conflicts.get(id(person), True) is not None
        conflicts = [c for c in person_conflicts.values() if c is not None]
        collision = min(conflicts, key=lambda c: c['time']) if conflicts else None
        emergency = (collision is not None and collision['time']
                     <= highway_parameters.MERGE_EMERGENCY_COLLISION_TIME)
        if emergency:
            pedestrian_target_velocity = 0.
        lattice_result.debug_details['jamming_pedestrian'] = dict(
            checked=len(person_conflicts), conflicts=len(conflicts),
            collision=collision, emergency=emergency)
    return person_check, pedestrian_target_velocity

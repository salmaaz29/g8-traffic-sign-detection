"""Classes GTSDB : 43 classes et regroupement en 4 catégories."""

CLASS_NAMES_43 = [
    "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60",
    "speed_limit_70", "speed_limit_80", "end_speed_limit_80", "speed_limit_100",
    "speed_limit_120", "no_passing", "no_passing_over_3.5t", "right_of_way",
    "priority_road", "yield", "stop", "no_vehicles", "no_vehicles_over_3.5t",
    "no_entry", "general_caution", "curve_left", "curve_right", "double_curve",
    "bumpy_road", "slippery_road", "road_narrows_right", "road_work",
    "traffic_signals", "pedestrians", "children_crossing", "bicycles_crossing",
    "ice_snow", "wild_animals", "end_all_limits", "turn_right_ahead",
    "turn_left_ahead", "ahead_only", "straight_or_right", "straight_or_left",
    "keep_right", "keep_left", "roundabout", "end_no_passing",
    "end_no_passing_over_3.5t",
]

CATEGORY_NAMES = ["prohibitory", "danger", "mandatory", "other"]

_PROHIBITORY = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
_DANGER = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
_MANDATORY = [33, 34, 35, 36, 37, 38, 39, 40]
_OTHER = [6, 12, 13, 14, 17, 32, 41, 42]

CLASS_TO_CATEGORY = {}
for _ids, _cat in ((_PROHIBITORY, 0), (_DANGER, 1), (_MANDATORY, 2), (_OTHER, 3)):
    for _i in _ids:
        CLASS_TO_CATEGORY[_i] = _cat
assert len(CLASS_TO_CATEGORY) == 43


def get_class_names(mode: str) -> list[str]:
    return CATEGORY_NAMES if mode == "4cls" else CLASS_NAMES_43


def map_class(class_id: int, mode: str) -> int:
    return CLASS_TO_CATEGORY[class_id] if mode == "4cls" else class_id

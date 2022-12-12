import json
import os

import numpy as np
from scipy import stats


def greater_stat_significant_value(task_errors: dict) -> float:
    """
    Compares the task errors to statistically infer the largest error and returns the
    corresponding p-value

    Alternative hypothesis: the max mean error is larger than all the others mean errors
    Null hypothesis: the max mean error is not larger than all the others mean errors

    We use the Welch-test to perform the hypothesis test

    :return: the p_value confirming the hypothesis
    """

    key_max_mean = None
    # max_mean initialized to lowest value it can get
    max_mean = 0
    # max_p_value initialized to lowest value it can get
    max_p_value = 0

    for key, value in task_errors.items():
        mean = np.mean(value)
        std = np.std(value)
        if key_max_mean is None or mean > max_mean:
            key_max_mean = key
            max_mean = mean
        task_errors[key] = (value, mean, std)

    for key, (value, mean, std) in task_errors.items():
        if key == key_max_mean:
            continue
        # compute p-value
        # Explanation of the formula for Welch test with distributions.
        # http://homework.uoregon.edu/pub/class/es202/ztest.html#:~:text=The%20simplest%20way%20to%20compare,is%20via%20the%20Z%2Dtest.&text=The%20error%20in%20the%20mean,mean%20value%20for%20that%20population.
        # noqa
        _, p_value = stats.ttest_ind(
            task_errors[key_max_mean][0], value, equal_var=False, alternative="greater",
        )
        if p_value > max_p_value:
            max_p_value = p_value
    return max_p_value


def smaller_stat_significant_value(task_errors: dict) -> float:
    """
    Compares the task errors to statistically infer the smallest error and returns the
    corresponding p-value

    Alternative hypothesis: the min mean error is smaller than all the others
                            mean errors
    Null hypothesis: the min mean error is not smaller than all the others mean errors

    We use the Welch-test to perform the hypothesis test

    :return: the p_value confirming the hypothesis
    """

    key_min_mean = None
    # min_mean initialized to highest value it can get
    min_mean = np.Inf
    # max_p_value initialized to lowest value it can get
    max_p_value = 0

    for key, value in task_errors.items():
        mean = np.mean(value)
        std = np.std(value)
        if key_min_mean is None or mean < min_mean:
            key_min_mean = key
            min_mean = mean
        task_errors[key] = (value, mean, std)

    for key, (value, mean, std) in task_errors.items():
        if key == key_min_mean:
            continue
        # compute p-value
        # Explanation of the formula for Welch test with distributions.
        # http://homework.uoregon.edu/pub/class/es202/ztest.html#:~:text=The%20simplest%20way%20to%20compare,is%20via%20the%20Z%2Dtest.&text=The%20error%20in%20the%20mean,mean%20value%20for%20that%20population.
        # noqa
        _, p_value = stats.ttest_ind(
            task_errors[key_min_mean][0], value, equal_var=False, alternative="less",
        )
        if p_value > max_p_value:
            max_p_value = p_value
    return max_p_value


def create_error_dict(errors: list[list[float]]) -> dict:

    task_errors = {}
    for n, error_list in enumerate(errors):
        task_errors[n + 1] = error_list

    return task_errors


if __name__ == "__main__":

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results_generation/performance_data/performance.json"
    )
    with open(data_path, 'r') as d:
        performance = json.load(d)
    dc_pos_errors = performance["door_closing_p_e"]
    do_pos_errors = performance["door_opening_p_e"]
    db_pos_errors = performance["deposit_brush_p_e"]
    bpu_pos_errors = performance["brush_picking_up_p_e"]
    rc_pos_error = performance["rail_cleaning_p_e"]

    list_errors = [
        do_pos_errors,
        bpu_pos_errors,
        rc_pos_error,
        db_pos_errors,
        dc_pos_errors,
    ]
    dict_errors = create_error_dict(list_errors)
    print(
        f"T3 has the largest e_p error with p-value of "
        f"{round(greater_stat_significant_value(dict_errors.copy()),3)}"
    )
    print(
        f"T4 has the smallest e_p error with p-value of "
        f"{round(smaller_stat_significant_value(dict_errors.copy()),7)}"
    )

    dc_ori_errors = performance["door_closing_o_e"]
    do_ori_errors = performance["door_opening_o_e"]
    db_ori_errors = performance["deposit_brush_o_e"]
    bpu_ori_errors = performance["brush_picking_up_o_e"]
    rc_ori_error = performance["rail_cleaning_o_e"]

    list_ori_errors = [
        do_ori_errors,
        bpu_ori_errors,
        rc_ori_error,
        db_ori_errors,
        dc_ori_errors,
    ]
    dict_ori_errors = create_error_dict(list_ori_errors)
    print(
        f"T3 has the largest e_theta error with p-value of "
        f"{round(greater_stat_significant_value(dict_ori_errors.copy()), 3)}"
    )
    print(
        f"T4 has the smallest e_theta error with p-value of "
        f"{round(smaller_stat_significant_value(dict_ori_errors.copy()), 7)}"
    )

    dc_j_errors = performance["door_closing_j_e"]
    do_j_errors = performance["door_opening_j_e"]
    db_j_errors = performance["deposit_brush_j_e"]
    bpu_j_errors = performance["brush_picking_up_j_e"]
    rc_j_error = performance["rail_cleaning_j_e"]

    list_j_errors = [
        do_j_errors,
        bpu_j_errors,
        rc_j_error,
        db_j_errors,
        dc_j_errors,
    ]
    dict_j_errors = create_error_dict(list_j_errors)
    print(
        f"T1 has the largest e_j error with p-value of "
        f"{round(greater_stat_significant_value(dict_j_errors.copy()), 3)}"
    )

    drill_pos_errors = performance["drill_p_e"]
    avoid_pos_error = performance["avoid_p_e"]
    parallel_pos_error = performance["parallel_p_e"]
    collab_pos_error = performance["collab_p_e"]

    list_f_pos_errors = [
        drill_pos_errors,
        avoid_pos_error,
        parallel_pos_error,
        collab_pos_error,
    ]

    dict_f_pos_errors = create_error_dict(list_f_pos_errors)
    print(
        f"F3 has the largest e_p error with p-value of "
        f"{round(greater_stat_significant_value(dict_f_pos_errors.copy()), 3)}"
    )

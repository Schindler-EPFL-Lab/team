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
    dc_pos_errors = [
        0.7443668450434872,
        0.7276571995109065,
        0.7352672983344076,
        0.7374788132550144,
        0.7105758228366948,
        0.7163469829628363,
        0.7132082444840285,
        0.7283543093852182,
        0.7247847956462455,
        0.7193476211123226,
    ]

    do_pos_errors = [
        0.8420866938741892,
        1.0608058257759034,
        1.5612270814971367,
        0.8399958333229938,
        1.0958471608760176,
        0.5758524116473345,
        1.2468043150390795,
        0.8885415015631393,
        1.053662659488215,
        1.4147586366585907,
    ]

    db_pos_errors = [
        0.002999999999985903,
        0.0022360679775104667,
        0.00707106781181483,
        0.008717797887096892,
        0.0031622776601565156,
        0.0036055512754654398,
        0.004690415759815528,
        0.004123105625616668,
        0.0036055512754575567,
        0.0028284271247395985,
    ]

    bpu_pos_errors = [
        0.8928280909559573,
        0.8903695861831638,
        0.8924057373191058,
        0.8907418256711511,
        0.902117508975425,
        0.9036653141511808,
        0.9013966940254321,
        0.8889651286749176,
        0.8968985449871162,
        0.8928297710090123,
    ]

    rc_pos_error = [
        0.9019556530118799,
        2.084589647868404,
        1.538803756169091,
        1.3745999418012382,
        1.9225948611186654,
        1.4329836007435797,
        1.11549540563828,
        1.1358173268620873,
        1.545111646451557,
        1.7399221246941192,
    ]

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

    dc_ori_errors = [
        0.0017839892790129197,
        0.0017867655774257735,
        0.0017911169697324729,
        0.0017983170509801528,
        0.0017912441706336116,
        0.0017999747528148666,
        0.0017886179058658222,
        0.0017917063155173996,
        0.0017941670812551232,
        0.0018086824009764942,
    ]

    do_ori_errors = [
        0.0010884528845944818,
        0.002569832298589272,
        0.00295857556545067,
        0.0018837795415762454,
        0.001242380525729199,
        0.001415510758353662,
        0.0020948902769515033,
        0.001589250423687834,
        0.002720471632781127,
        0.0024208806286670106,
    ]

    db_ori_errors = [
        1.616130869592985e-05,
        8.624939930758328e-06,
        1.594130362296972e-05,
        1.644688416941544e-05,
        5.836416059319239e-06,
        1.2326755110893656e-05,
        1.4468641008117848e-05,
        2.64849642037234e-05,
        9.401444778367576e-06,
        7.116326428461325e-06,
    ]

    bpu_ori_errors = [
        0.00403120604862502,
        0.004028291111955753,
        0.004033306855995963,
        0.00403407064816182,
        0.0041177203089174585,
        0.0041307901143041136,
        0.00412404691434014,
        0.004150247845805909,
        0.004172751811635773,
        0.004168126259563874,
    ]

    rc_ori_error = [
        0.00410123642542732,
        0.005374554305835034,
        0.004843792004396907,
        0.004830981981392911,
        0.008712098955699831,
        0.007881608610108816,
        0.003286479823113418,
        0.003052182400210195,
        0.006044414086447588,
        0.006373101348342394,
    ]

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

    dc_j_errors = [
        0.29898885206583103,
        0.2933713894531958,
        0.5391626532578065,
        0.44057215563048563,
        0.5441634147650718,
        0.5464943587364667,
        0.30292883084565503,
        0.34651992112528995,
        0.27367432371144407,
        0.29970391631407484,
    ]

    do_j_errors = [
        0.34473088507734495,
        0.30515511914877,
        1.007851046624798,
        0.20960273891428058,
        0.8659363097200297,
        0.6940808669162556,
        0.9795139185615906,
        1.1617438448607724,
        1.0873516331933615,
        0.8593347700145945,
    ]

    db_j_errors = [
        0.38520734413013796,
        0.42844275510002133,
        0.4183572859012023,
        0.4100910799403508,
        0.40654851933933,
        0.6863959464255341,
        0.42550286542373333,
        0.5821284220130727,
        0.5543843038663817,
        0.3854858433575366,
    ]

    bpu_j_errors = [
        0.32913464140719206,
        0.35372257324589984,
        0.2582731133893741,
        0.9066635051960704,
        0.45134998815541005,
        0.355950927526738,
        0.5226738562584786,
        0.6117366667864098,
        0.4399065167685608,
        0.5769654596267394,
    ]

    rc_j_error = [
        0.39892966189835355,
        0.3221201673836503,
        0.4400144246290704,
        0.43111297330404846,
        0.3245745762255436,
        0.594105041078773,
        0.5050451084529155,
        0.1969249393442604,
        0.235661208757053,
        0.4635870957666807,
    ]

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

    drill_pos_errors = [
        0.050717255446266186,
        0.04670160596810882,
        0.04670160596810882,
        0.050717255446266186,
        0.04670160596810882,
        0.04670160596810882,
        0.04670160596810882,
        0.04670160596810882,
        0.04670160596810882,
        0.04670160596810882,
        0.04670160596810882,
        0.027033497738902436,
        0.0262276571580612,
        0.024340090386021283,
        0.024340090386021283,
        0.017895530168167806,
        0.017895530168167806,
        0.0262276571580612,
        0.0262276571580612,
        0.017895530168167806,
        0.017895530168167806,
        0.0,
        0.01041921302209206,
        0.0,
        0.004199999999997317,
        0.005418486873636606,
        0.012411285187297725,
        0.0036891733491319976,
        0.0036891733491319976,
        0.0036891733491319976,
        0.01041921302209206,
    ]

    avoid_pos_error = [
        0.17846837254819703,
        0.19542968044797282,
        0.17846837254819703,
        0.17846837254819703,
        0.17846837254819703,
        0.17846837254819703,
        0.17846837254819703,
        0.19542968044797282,
        0.19542968044797282,
        0.19542968044797282,
        0.12103838234215411,
        0.12290711940324704,
        0.12103838234215411,
        0.12103838234215411,
        0.12290711940324704,
        0.12290711940324704,
        0.12290711940324704,
        0.12290711940324704,
        0.12290711940324704,
        0.12290711940324704,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
        0.1508628847662786,
    ]

    parallel_pos_error = [
        1.2602473725424188,
        1.2736412721013877,
        1.2736412721013877,
        1.2736412721013877,
        1.2736412721013877,
        1.330374323263951,
        1.2602473725424188,
        1.4903579972610534,
        1.4903579972610534,
        1.4903579972610534,
        1.4903579972610534,
        1.4903579972610534,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.004031128874112119,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
        0.07280109889277334,
    ]

    collab_pos_error = [
        0.5921503187536157,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.5863147618813552,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
        0.00318747549014435,
    ]

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

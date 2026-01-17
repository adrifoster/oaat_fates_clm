"""Data processing methods"""

import glob
import functools
import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date


def create_target_grid(file: str, var: str) -> xr.Dataset:
    """Creates a target grid to resample to

    Args:
        file (str): path to dataset to regrid to
        var (str): variable to create the grid off of

    Returns:
        xr.Dataset: output dataset
    """

    ds = xr.open_dataset(file)
    target_grid = ds[var].mean(dim="time")
    target_grid["area"] = ds["area"].fillna(0)
    target_grid["landmask"] = ds["landmask"].fillna(0)
    target_grid["landfrac"] = ds["landfrac"].fillna(0)
    target_grid["land_area"] = target_grid.area * target_grid.landfrac
    target_grid["land_area"] = target_grid["land_area"].where(
        target_grid.lat > -60.0, 0.0
    )

    return target_grid


def get_mesh_points(mesh: xr.Dataset):

    mesh = mesh.where(mesh.elementMask == 1, drop=True)
    centerCoords = mesh.centerCoords.values
    mesh_lats = [coord[1] for coord in centerCoords]
    mesh_lons = [coord[0] for coord in centerCoords]

    return mesh_lats, mesh_lons


def get_clm_param_dat(param_info_file, param_key, to_xarray=True):

    param_dat = (
        pd.read_csv(param_info_file, index_col=[0])
        .drop(columns=["min", "max", "location"])
        .drop_duplicates()
    )
    param_dat.columns = ["parameter_name", "long_name", "category", "subcategory"]

    param_info = pd.merge(
        param_dat,
        param_key,
        on="parameter_name",
    )
    param_info.ensemble = [
        int(str(e).replace("CLM6SPoaat", "")) for e in param_info.ensemble
    ]
    if to_xarray:
        param_info = param_info.set_index("ensemble").to_xarray()

    return param_info


def get_fates_param_dat(
    fates_param_list_file: str, oaat_key: pd.DataFrame, to_xarray=True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns pandas DataFrames with information about FATES parameters associated with a
    one-at-a-time ensemble

    Args:
        fates_param_list_file (str): path to FATES parameter list file (excel)
        oaat_key (pd.DataFrame): one-at-a-time ensemble key
        to_xarray (optional, bool): whether or not to converto xarray dataset. Defaults to True

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: data about all parameters and just those
        associated with the ensemble
    """

    # information about the parameters - only ones we can calibrate
    param_dat = pd.read_excel(fates_param_list_file)
    param_dat = param_dat[param_dat["calibrate"] == "Y"]

    # fix this - we called it 'fates_nonhydro_smpsc' in the oaat key
    param_dat["fates_parameter_name"] = param_dat["fates_parameter_name"].replace(
        {"smpsc_delta": "fates_nonhydro_smpsc"}
    )

    # merge with key
    param_info = pd.merge(
        param_dat[["fates_parameter_name", "long_name", "category", "subcategory"]],
        oaat_key,
        left_on="fates_parameter_name",
        right_on="parameter_name",
    )
    param_info = param_info.drop(columns=["fates_parameter_name"])

    if to_xarray:
        param_info = param_info.set_index("ensemble").to_xarray()

    return param_info


def get_all_parameters(clm_param_dat, fates_param_dat):
    clm_param = (
        clm_param_dat.to_pandas()
        .reset_index()
        .drop(columns=["type", "ensemble"])
        .drop_duplicates()
    )
    clm_param["model"] = "CLM"

    fates_param = (
        fates_param_dat.to_pandas()
        .reset_index()
        .drop(columns=["type", "ensemble"])
        .drop_duplicates()
    )
    fates_param["model"] = "FATES"

    return pd.concat([clm_param, fates_param])


def count_parameters(param_key, exclude_pars=None):
    param_key = param_key[param_key.type != "default"]
    if exclude_pars is not None:
        param_key = param_key[~param_key.parameter_name.isin(exclude_pars)]
    return len(param_key.parameter_name.unique())


def count_if_PFT_independent(param_key, param_dat, exclude_pars=None, FATES=True):

    param_key = param_key[param_key.type != "default"]

    if exclude_pars is not None:
        param_key = param_key[~param_key.parameter_name.isin(exclude_pars)]

    params = param_key.parameter_name.unique()

    pft_dim = "fates_pft" if FATES else "pft"

    pft_params = []
    global_params = []
    for parameter in params:
        if parameter in param_dat.data_vars:
            if pft_dim in param_dat[parameter].dims:
                pft_params.append(parameter)
            else:
                global_params.append(parameter)
        else:
            global_params.append(parameter)

    return len(pft_params) * 16 + len(global_params)


def get_area_means_diffs(
    file: str,
    param_info: xr.Dataset,
    out_vars: list[str],
    default_ind: int = 0,
    remove_vars: list[str] = None,
) -> xr.Dataset:
    """Gets the sum of all differences between mean and iav for across all history variables
    for each ensemble member

    Args:
        file (str): path to ensemble dataset
        param_info (xr.Datset): data frame with information about parameters
        out_vars (list[str]): list of output variables
        default_ind (int, optional): index of default simulation. Defaults to 0.
        remove_vars (list[str], optional): list of variables to remove from ensemble. Defaults to None.

    Returns:
        xr.Dataset: output dataset with differences
    """

    ds = xr.open_dataset(file)
    ds["WUE"] = ds["GPP"] / ds["QVEGT"].where(ds.QVEGT > 0.0)
    default_mean = ds.sel(ensemble=default_ind).sel(summation_var="mean")
    default_iav = ds.sel(ensemble=default_ind).sel(summation_var="iav")
    mean_vals = ds.sel(summation_var="mean")
    iav_vals = ds.sel(summation_var="iav")

    mean_diffs = get_differences(mean_vals, out_vars, default_mean)
    mean_iavs = get_differences(iav_vals, out_vars, default_iav)

    mean_sum_diff = mean_diffs.sum(dim="variable")
    mean_iav_diff = mean_iavs.sum(dim="variable")

    ds["sum_diff"] = mean_sum_diff + mean_iav_diff
    ds = xr.merge([ds, param_info], join="outer")

    if remove_vars is not None:
        ds = ds.where(~ds.parameter_name.isin(remove_vars), drop=True)

    ds_mean = ds.sel(summation_var="mean")
    ds_iav = ds.sel(summation_var="iav")

    return ds, ds_mean, ds_iav


def get_differences(
    ds: xr.Dataset, out_vars: list[str], default: xr.Dataset
) -> xr.Dataset:
    """Gets differences between the default and the ensemble member for all input variables

    Args:
        ds (xr.Dataset): ensemble dataset
        out_vars (list[str]): list of variables to compare
        default (xr.Dataset): default ensemble member

    Returns:
        xr.Dataset: output difference dataset
    """

    diff_dfs = []
    for variable in out_vars:
        diff = np.abs(ds[variable] - default[variable])
        diff.name = "absolute_difference"
        diff_dfs.append(diff)

    diff = xr.concat(diff_dfs, dim="variable")
    diff = diff.assign_coords(variable=("variable", out_vars))

    return diff


def get_combined(ds1, ds2, name1, name2):

    ds1 = ds1.assign(sim_source=("ensemble", [name1] * ds1.sizes["ensemble"]))
    ds2 = ds2.assign(sim_source=("ensemble", [name2] * ds2.sizes["ensemble"]))

    ds2_shifted = ds2.assign_coords(ensemble=ds2.ensemble + ds1.sizes["ensemble"])

    return xr.concat([ds1, ds2_shifted], dim="ensemble")


def get_active_ensemble_df(clm_ds, fates_ds):
    clm_active_ens = clm_ds.where(clm_ds.sum_diff > 0.0, drop=True)
    clm_active_ens = clm_active_ens.to_pandas().reset_index().drop(columns=["ensemble"])
    clm_active_ens["model"] = "CLM"

    fates_active_ens = fates_ds.where(fates_ds.sum_diff > 0.0, drop=True)
    fates_active_ens = (
        fates_active_ens.to_pandas().reset_index().drop(columns=["ensemble"])
    )
    fates_active_ens["model"] = "FATES"

    return pd.concat([clm_active_ens, fates_active_ens])


def get_params(fates_ds, fates_clm_ds, clm_ds, var="sum_diff"):

    fates_fates_nonzero = get_nonzero_params(fates_ds, var=var)
    fates_clm_nonzero = get_nonzero_params(fates_clm_ds, var=var)
    clm_nonzero = get_nonzero_params(clm_ds, var=var)

    all_nonzero = np.unique(
        np.append(np.append(fates_fates_nonzero, fates_clm_nonzero), clm_nonzero)
    )

    fates_only_parameters = np.append(
        [param for param in fates_clm_nonzero if param not in clm_nonzero],
        fates_fates_nonzero,
    )

    clm_only_parameters = [
        param for param in clm_nonzero if param not in fates_clm_nonzero
    ]
    shared_parameters = [param for param in clm_nonzero if param in fates_clm_nonzero]

    out_dict = {
        "fates": fates_fates_nonzero,
        "fates_clm": fates_clm_nonzero,
        "clm": clm_nonzero,
        "clm_only": clm_only_parameters,
        "fates_only": fates_only_parameters,
        "common": shared_parameters,
        "all_nonzero": all_nonzero,
    }

    return out_dict


def get_nonzero_params(ds, var="sum_diff"):
    return np.unique(ds.where(ds[var] > 0.0, drop=True).parameter_name.values)


def classify_params(all_params, nonzero_params):

    clm_only_params = all_params[
        all_params.parameter_name.isin(nonzero_params["clm_only"])
    ].copy()
    clm_only_params["type"] = "CLM only"

    fates_only_params = all_params[
        all_params.parameter_name.isin(nonzero_params["fates_only"])
    ].copy()
    fates_only_params["type"] = "FATES only"

    common_params = all_params[
        all_params.parameter_name.isin(nonzero_params["common"])
    ].copy()
    common_params["type"] = "common"

    return pd.concat([clm_only_params, fates_only_params, common_params])


def get_pct_diff(active_df, variable, default_ds, tol=1.0):
    default_value = default_ds[variable].values
    all_len = len(active_df.parameter_name.unique())
    active_df["pct_diff"] = np.abs(
        (active_df[variable] - default_value) / default_value * 100
    )

    above_tol = active_df[active_df.pct_diff > tol]
    params = above_tol.parameter_name.unique()
    num_above = len(params)

    return num_above / all_len * 100.0, num_above


def get_param_variance(parameters, variable, ds, default_ind):

    default = ds.isel(ensemble=default_ind)

    variances = []
    for parameter in parameters:

        this_par = ds.where(ds.parameter_name == parameter, drop=True)

        if (this_par.type == "min").any():
            min_par = this_par.where(this_par.type == "min", drop=True)
        else:
            min_par = default

        if (this_par.type == "max").any():
            max_par = this_par.where(this_par.type == "max", drop=True)
        else:
            max_par = default

        variance = (default[variable].values - min_par[variable].values) ** 2 + (
            max_par[variable].values - default[variable].values
        ) ** 2
        variances.append(variance[0])

    return pd.DataFrame({"parameter_name": parameters, "variance": variances})


def get_cumulative_variance(df, parameters, param_chunks):

    df = df.set_index("parameter_name").to_xarray()
    chunks = xr.DataArray(
        param_chunks
        + param_chunks * np.floor(np.arange(len(parameters)) / param_chunks),
        dims="parameter_name",
        name="nparams",
    )
    return (
        df["variance"]
        .sortby(df["variance"], ascending=False)
        .groupby(chunks)
        .sum()
        .cumsum(dim="nparams")
        / df["variance"].sum()
    )


def get_all_cumulative_variance(
    variables, clm_pars, clm_glob, fates_pars, fates_glob, n=5
):
    variances = {}
    for variable in variables:
        variances[variable] = {}

        clm_var = get_param_variance(clm_pars, variable, clm_glob, 0)
        variances[variable]["CLM"] = get_cumulative_variance(clm_var, clm_pars, n)

        fates_var = get_param_variance(fates_pars, variable, fates_glob, 0)
        variances[variable]["FATES"] = get_cumulative_variance(fates_var, fates_pars, n)

    return variances


def find_cumulative_params(params, variable, df, cutoff=0.9):
    variance_df = get_param_variance(params, variable, df, 0)
    variance_df = variance_df.sort_values(by="variance", ascending=False).reset_index()
    variance_df["cum_sum"] = variance_df.variance.cumsum() / variance_df.variance.sum()

    mask = variance_df.cum_sum >= cutoff
    first_true = mask.idxmax()
    return variance_df.iloc[: (first_true + 1)]["parameter_name"].values


def get_min_max(df, variable):
    var_df = df[df.variable == variable]
    return var_df.min_value.values[0], var_df.max_value.values[0]


def get_ensemble_ranges(ensemble_df, vars):
    mean_vals = {}
    max_vals = {}
    min_vals = {}
    diff = {}
    stds = {}
    variance = {}
    q1s = {}
    q3s = {}
    for variable in vars:
        mean_vals[variable] = ensemble_df[variable].mean()
        max_vals[variable] = ensemble_df[variable].max()
        min_vals[variable] = ensemble_df[variable].min()
        stds[variable] = ensemble_df[variable].std()
        variance[variable] = ensemble_df[variable].var()
        diff[variable] = np.abs(
            ensemble_df[variable].max() - ensemble_df[variable].min()
        )
        q1s[variable] = ensemble_df[variable].quantile(0.25)
        q3s[variable] = ensemble_df[variable].quantile(0.75)

    df = pd.DataFrame(
        {
            "mean": mean_vals,
            "max": max_vals,
            "min": min_vals,
            "range": diff,
            "std": stds,
            "variance": variance,
            "q1": q1s,
            "q3": q3s,
        }
    )
    df["variable"] = df.index
    df["CV"] = df["std"] / df["mean"]
    df["iqr"] = df["q3"] - df["q1"]

    return df


def print_ensemble_range(df, model, variable, units):

    var_df = df[df.variable == variable]
    mod_df = var_df[var_df.model == model]

    print(
        f"{model} {variable} ranges from",
        round(mod_df["min"].values[0], 2),
        "to",
        round(mod_df["max"].values[0], 2),
        units,
    )

    print("This is a range of ", round(mod_df["range"].values[0], 2), units)
    print("And a mean of ", round(mod_df["mean"].values[0], 2), units)
    print("And a standard devaiation of ", round(mod_df["std"].values[0], 2), units)
    print("And a variance of ", round(mod_df["variance"].values[0], 2), units)
    print("And an IQR of ", round(mod_df["iqr"].values[0], 2), units)


def get_both_ranges(active_df, vars):

    fates_ensemble = active_df[active_df.model == "FATES"]
    clm_ensemble = active_df[active_df.model == "CLM"]

    fates_df = get_ensemble_ranges(fates_ensemble, vars)
    fates_df["model"] = "FATES"

    clm_df = get_ensemble_ranges(clm_ensemble, vars)
    clm_df["model"] = "CLM"

    range_df = pd.concat([fates_df, clm_df])

    mean_df = np.abs(range_df.groupby("variable")["mean"].mean())

    range_df["range_norm"] = range_df.apply(
        lambda row: row["range"] / mean_df[row["variable"]], axis=1
    )

    return range_df


def get_min_max_diff(ds: xr.Dataset, model: str) -> pd.DataFrame:
    """Gets differences between min and max ensemble members for all variables

    Args:
        ds (xr.Dataset): ensemble dataset
        sumvar (str): summation variable ['mean', 'iav']

    Returns:
        pd.DataFrame: output dataframe
    """

    # we don't want to look at these data variables
    skip_vars = [
        "parameter_name",
        "type",
        "category",
        "subcategory",
        "sum_diff",
        "sim_source",
        "long_name",
    ]
    vars_to_check = [v for v in ds.data_vars if v not in skip_vars]

    default_ds = ds.where(ds.ensemble == 0, drop=True)

    # group by parameter name
    grouped = ds.groupby("parameter_name")
    diffs = {}
    for param, group in grouped:
        # select the min and max rows
        if (group.type == "min").any():
            min_val = group.where(group.type == "min", drop=True)
        else:
            min_val = default_ds

        # Check if 'max' exists in the group
        if (group.type == "max").any():
            max_val = group.where(group.type == "max", drop=True)
        else:
            max_val = default_ds

        # sanity check: if either is missing, skip
        if min_val.sizes["ensemble"] == 0 or max_val.sizes["ensemble"] == 0:
            continue

        # assume one row per type per parameter
        min_val = min_val.isel(ensemble=0)
        max_val = max_val.isel(ensemble=0)

        # compute differences for each variable
        diffs[param] = {}
        for var in vars_to_check:
            diffs[param][var] = np.abs((max_val[var] - min_val[var])).item()

    df_diffs = pd.DataFrame.from_dict(diffs, orient="index")
    df_diffs.index.name = "parameter_name"
    df_diffs["parameter"] = df_diffs.index
    df_diffs["model"] = model

    return df_diffs


def get_top_n(
    ds: xr.Dataset,
    df_diffs: pd.DataFrame,
    variable: str,
    n: int,
    default_ds,
    exclude_list=None,
) -> pd.DataFrame:
    """Gets the top n ensemble members with the most impact on variable

    Args:
        ds (xr.Dataset): ensemble dataset
        df_diffs (pd.DataFrame): difference data frame
        variable (str): variable name
        n (int): number to include
        sumvar (str): summation variable ['mean' or 'iav']

    Returns:
        pd.DataFrame: output data frame
    """

    # get top n parameters for this variable
    if exclude_list is not None:
        df_diffs = df_diffs.loc[~df_diffs.index.isin(exclude_list)]
    top_params = df_diffs[variable].sort_values(ascending=False).head(n).index

    results = []
    for param in top_params:
        sub = ds.where(ds.parameter_name == param, drop=True)

        if (sub.type == "min").any():
            min_run = sub.where(sub.type == "min", drop=True).isel(ensemble=0)
            category = min_run["category"].item()
            subcategory = min_run["subcategory"].item()
        else:
            min_run = default_ds

        # Check if 'max' exists in the group
        if (sub.type == "max").any():
            max_run = sub.where(sub.type == "max", drop=True).isel(ensemble=0)
            category = max_run["category"].item()
            subcategory = max_run["subcategory"].item()
        else:
            max_run = default_ds

        results.append(
            {
                "parameter_name": param,
                "min_val": min_run[variable].item(),
                "max_val": max_run[variable].item(),
                "default": default_ds[variable].item(),
                "difference": max_run[variable].item() - min_run[variable].item(),
                "category": category,
                "subcategory": subcategory,
            }
        )
    return pd.DataFrame(results)


def get_biome_df(biome_ds, model):
    biomes = biome_ds.biome.values
    biome_diffs = []
    for biome in biomes:
        df = get_min_max_diff(biome_ds.sel(biome=biome), model)
        df["biome"] = biome
        biome_diffs.append(df)
    return pd.concat(biome_diffs)


def get_biome_top_n(biome_ds, biome_df, variable, n=10):
    biomes = biome_ds.biome.values
    topns = []
    for biome in biomes:
        biome_mean = biome_ds.sel(biome=biome)
        diff_df = biome_df[biome_df.biome == biome]
        top_n = get_top_n(biome_mean, diff_df, variable, n, biome_mean.sel(ensemble=0))
        top_n["biome"] = biome
        topns.append(top_n)
    return pd.concat(topns)


def get_vardiff(da, baseline_dat, variables, params, n, reldiff=False, include_sd=True):

    all_var_dfs = {}
    all_var_sd_dfs = {}
    for variable in variables:
        var_diffs = {}
        var_sds = {}

        for param in params:
            var_diffs[param] = {}
            var_sds[param] = {}

            dat = da.where(da.parameter_name == param, drop=True)

            if (dat.type == "min").any():
                var_min = dat.where(dat.type == "min", drop=True)
            else:
                var_min = da.isel(ensemble=0)

            if (dat.type == "max").any():
                var_max = dat.where(dat.type == "max", drop=True)
            else:
                var_max = da.isel(ensemble=0)

            if reldiff:
                var_diff = (
                    (
                        var_max.sel(summation_var="mean")[variable].values
                        - var_min.sel(summation_var="mean")[variable].values
                    )
                    / baseline_dat.sel(summation_var="mean")[variable].values
                    * 100.0
                )
                sd_diff = (
                    np.sqrt(
                        var_max.sel(summation_var="iav")[variable].values / n
                        + var_min.sel(summation_var="iav")[variable].values / n
                    )
                    / baseline_dat.sel(summation_var="mean")[variable].values
                    * 100.0
                )
            else:
                var_diff = (
                    var_max.sel(summation_var="mean")[variable].values
                    - var_min.sel(summation_var="mean")[variable].values
                )
                sd_diff = np.sqrt(
                    var_max.sel(summation_var="iav")[variable].values / n
                    + var_min.sel(summation_var="iav")[variable].values / n
                )

            diff = np.atleast_1d(var_diff)[0]
            diff_sd = np.atleast_1d(sd_diff)[0]
            var_diffs[param][variable] = diff
            var_sds[param][variable] = diff_sd

        var_df = pd.DataFrame.from_dict(var_diffs, orient="index")
        var_sd_df = pd.DataFrame.from_dict(var_sds, orient="index")
        all_var_dfs[variable] = var_df
        all_var_sd_dfs[variable] = var_sd_df

    mean_df = pd.concat(all_var_dfs.values(), axis=1)
    sd_df = pd.concat(all_var_sd_dfs.values(), axis=1)

    if include_sd:
        return pd.merge(
            mean_df, sd_df, left_index=True, right_index=True, suffixes=("_mean", "_sd")
        )
    else:
        return mean_df


def get_all_vardiffs(
    variables, clm_ds, fatesclm_ds, fates_ds, nonzero_params, n, reldiff=False
):

    clm_diffs = get_vardiff(
        clm_ds,
        clm_ds.sel(ensemble=0),
        variables,
        nonzero_params["clm"],
        n,
        reldiff=reldiff,
    )
    clm_diffs["model"] = "CLM"
    clm_diffs["parameter"] = clm_diffs.index

    fatesclm_diffs = get_vardiff(
        fatesclm_ds,
        fatesclm_ds.sel(ensemble=0),
        variables,
        nonzero_params["fates_clm"],
        n,
        reldiff=reldiff,
    )
    fatesclm_diffs["model"] = "FATES"
    fatesclm_diffs["parameter"] = fatesclm_diffs.index

    fates_diffs = get_vardiff(
        fates_ds,
        fates_ds.sel(ensemble=0),
        variables,
        nonzero_params["fates"],
        n,
        reldiff=reldiff,
    )
    fates_diffs["model"] = "FATES"
    fates_diffs["parameter"] = fates_diffs.index

    clm_sub = clm_diffs[clm_diffs.index.isin(fatesclm_diffs.parameter)]
    fates_clm_diff = pd.concat([fatesclm_diffs, clm_sub])
    fates_diffs = pd.concat([fatesclm_diffs, fates_diffs])

    return clm_diffs, fates_diffs, fates_clm_diff


def get_compare_df(clm_reldiffs, fates_reldiffs, clm_parameters, fates_parameters):

    clm_sub = (
        clm_reldiffs[clm_reldiffs.parameter.isin(clm_parameters)]
        .reset_index()
        .drop(columns=["index"])
    )
    fates_sub = (
        fates_reldiffs[fates_reldiffs.parameter.isin(fates_parameters)]
        .reset_index()
        .drop(columns=["index"])
    )

    both_sub = pd.concat([clm_sub, fates_sub]).melt(id_vars=["model", "parameter"])
    both_sub[["base_var", "stat"]] = both_sub["variable"].str.extract(r"(.*)_(mean|sd)")
    df_wide = both_sub.pivot_table(
        index=["model", "base_var", "parameter"], columns="stat", values="value"
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(
        columns={"mean": "mean_value", "sd": "sd_value", "base_var": "variable"}
    )

    return df_wide


def get_slope(df, varx, vary, model, category):

    df_model = df[df["model_name"] == model]
    df_cat = df_model[df_model.category_subset == category]
    x = df_cat[varx]
    y = df_cat[vary]
    slope, _, _, _, _ = stats.linregress(x, y)

    return slope


def create_combined_mini_oaat_data(
    variable,
    fates_glob_combo_mean2,
    fates_meandiffs2,
    fates_glob_combo_mean3,
    fates_meandiffs3,
    fates_glob_combo_mean,
    fates_meandiffs_sub,
    clm_mean,
    clm_meandiffs_sub,
    corresponding_params,
):

    fates_top10_2 = get_top_n(
        fates_glob_combo_mean2,
        fates_meandiffs2,
        variable,
        10,
        fates_glob_combo_mean2.sel(ensemble=0),
    )
    fates_top10_2["version"] = "CLM-FATES parameter update"
    fates_top10_3 = get_top_n(
        fates_glob_combo_mean3,
        fates_meandiffs3,
        variable,
        10,
        fates_glob_combo_mean3.sel(ensemble=0),
    )
    fates_top10_3["version"] = "CLM-FATES parameter & water stress update"
    fates_top10_sub = get_top_n(
        fates_glob_combo_mean,
        fates_meandiffs_sub,
        variable,
        10,
        fates_glob_combo_mean.sel(ensemble=0),
    )
    fates_top10_sub["version"] = "CLM-FATES standard configuration"
    clm_top10_sub = get_top_n(
        clm_mean, clm_meandiffs_sub, variable, 10, clm_mean.sel(ensemble=0)
    )
    clm_top10_sub["version"] = "CLM"

    all_top = pd.concat([fates_top10_2, fates_top10_3, fates_top10_sub, clm_top10_sub])
    all_top["analagous_parameter"] = (
        all_top["parameter_name"]
        .map(corresponding_params)
        .fillna(all_top["parameter_name"])
    )

    return all_top


def get_clm_ds(
    files: list[str],
    data_vars: list[str],
    start_year: int,
    run_dict: dict = None,
) -> xr.Dataset:
    """Reads in a CLM dataset and does some initial post-processing

    Args:
        files (list[str]): list of files
        data_vars (list[str]): data variables to read in
        start_year (int): start year
        run_dict (dict, optional): Dictionary describing aspects of the run:
            fates (bool, optional): is it a FATES run? defaults to True.
            sparse (bool, optional): is it a sparse run? Defaults to True.
            ensemble (int, optional): ensemble member. Defaults to None

    Returns:
        xr.Dataset: output dataset
    """

    # create an empty dictionary if not supplied
    if run_dict is None:
        run_dict = {}

    # read in dataset
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=functools.partial(preprocess, data_vars=data_vars),
        parallel=True,
        autoclose=True,
        decode_timedelta=True,
    )

    # update time
    ds["time"] = xr.cftime_range(str(start_year), periods=len(ds.time), freq="MS")

    if run_dict.get("fates", True):
        ds["GPP"] = ds["FATES_GPP"] * ds["FATES_FRACTION"]  # kg m-2 s-1
        ds["GPP"].attrs["units"] = ds["FATES_GPP"].attrs["units"]
        ds["GPP"].attrs["long_name"] = ds["FATES_GPP"].attrs["long_name"]

        ds["LAI"] = ds["FATES_LAI"] * ds["FATES_FRACTION"]  # m m-2
        ds["LAI"].attrs["units"] = ds["FATES_LAI"].attrs["units"]
        ds["LAI"].attrs["long_name"] = ds["FATES_LAI"].attrs["long_name"]

    else:
        ds["GPP"] = ds["FPSN"] * 1e-6 * 12.011 / 1000.0  # kg m-2 s-1
        ds["GPP"].attrs["units"] = "kg m-2 s-1"
        ds["GPP"].attrs["long_name"] = ds["FPSN"].attrs["long_name"]

        ds["LAI"] = ds["TLAI"]  # m m-2
        ds["LAI"].attrs["units"] = ds["TLAI"].attrs["units"]
        ds["LAI"].attrs["long_name"] = ds["TLAI"].attrs["long_name"]

    sh = ds.FSH
    le = ds.EFLX_LH_TOT
    energy_threshold = 20

    sh = sh.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    le = le.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    ds["EF"] = le / (le + sh)
    ds["EF"].attrs["units"] = "unitless"
    ds["EF"].attrs["long_name"] = "Evaporative fraction"

    rsds = ds.FSDS.where(ds.FSDS >= 10)
    rsus = ds.FSR.where(ds.FSDS >= 10)
    ds["ASA"] = rsus / rsds
    ds["ASA"].attrs["units"] = "unitless"
    ds["ASA"].attrs["long_name"] = "All sky albedo"

    ds["RLNS"] = ds.FLDS - ds.FIRE
    ds["RLNS"].attrs["units"] = "W m-2"
    ds["RLNS"].attrs["long_name"] = "surface net longwave radiation"

    ds["RN"] = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
    ds["RN"].attrs["units"] = "W m-2"
    ds["RN"].attrs["long_name"] = "surface net radiation"

    ds["Temp"] = ds.TSA - 273.15
    ds["Temp"].attrs["units"] = "degrees C"
    ds["Temp"].attrs["long_name"] = ds["TSA"].attrs["long_name"]

    ds["Precip"] = ds.SNOW + ds.RAIN
    ds["Precip"].attrs["units"] = "mm s-1"
    ds["Precip"].attrs["long_name"] = "total precipitation"

    ds["ET"] = ds.QVEGE + ds.QVEGT + ds.QSOIL
    ds["ET"].attrs["units"] = ds["QVEGE"].attrs["units"]
    ds["ET"].attrs["long_name"] = "evapotranspiration"

    ds["DTR"] = ds.TREFMXAV - ds.TREFMNAV
    ds["DTR"].attrs["units"] = ds["TREFMXAV"].attrs["units"]
    ds["DTR"].attrs["long_name"] = "diurnal temperature range"

    if run_dict.get("sparse", True):
        ds0 = xr.open_dataset(files[0], decode_timedelta=True)
        extras = ["grid1d_lat", "grid1d_lon"]
        for extra in extras:
            ds[extra] = ds0[extra]

    if run_dict.get("ensemble", None) is not None:
        ds["ensemble"] = run_dict["ensemble"]

    ds.attrs["Date"] = str(date.today())
    ds.attrs["Original"] = files[0]

    return ds


def post_process_ds(
    hist_dir: str,
    data_vars: list[str],
    years: list[int],
    run_dict: dict = {},
    whittaker_ds: xr.Dataset = None,
) -> xr.Dataset:
    """Post-processes a CLM dataset

    Args:
        hist_dir (str): history directory
        data_vars (list[str]): history variables to read in
        whittaker_ds (xr.Dataset): Whittaker biome dataset
        years (list[int]): start and end year of simulation
        run_dict (dict, optional): Dictionary describing aspects of the run:
            fates (bool, optional): is it a FATES run? defaults to True.
            sparse (bool, optional): is it a sparse run? Defaults to True.
            ensemble (int, optional): ensemble member. Default to None.
            filter_nyears (int, optional): How many years to filter at end of simulation.
                Defaults to None.

    Returns:
        xr.Dataset: output dataset
    """

    # assign default values if not there
    sparse = run_dict.get("sparse", True)
    filter_years = run_dict.get("filter_years", None)

    # read in dataset and calculate/convert units on some variables
    ds = get_clm_ds(
        get_files(hist_dir),
        data_vars,
        years[0],
        run_dict,
    )

    # add Whittaker biomes if we are doing a "sparse" run
    if sparse and whittaker_ds is not None:
        ds["biome"] = whittaker_ds.biome
        ds["biome_name"] = whittaker_ds.biome_name

    # filter on years
    if filter_years is not None:
        ds = ds.sel(time=slice(f"{filter_years[0]}-01-01", f"{filter_years[-1]}-12-31"))
        ds["time"] = xr.cftime_range(str(years[0]), periods=len(ds.time), freq="MS")

    return ds


def get_files(hist_dir: str, hstream="h0") -> list[str]:
    """Returns all CLM history files in a directory given an input hstream

    Args:
        hist_dir (str): directory
        hstream (str, optional): history level. Defaults to 'h0'.

    Returns:
        list[str]: list of files
    """
    return sorted(glob.glob(f"{hist_dir}/*clm2.{hstream}*.nc"))


def preprocess(data_set: xr.Dataset, data_vars: list[str]) -> xr.Dataset:
    """Preprocesses an xarray Dataset by subsetting to specific variables - to be used on read-in

    Args:
        data_set (xr.Dataset): input Dataset

    Returns:
        xr.Dataset: output Dataset
    """

    return data_set[data_vars]


def apply_to_vars(
    ds: xr.Dataset, varlist: list[str], func, add_sparse: bool, *args, **kwargs
) -> xr.Dataset:
    """Applies a function to each variable in varlist and merges results.

    Args:
        ds (xr.Dataset): Input dataset.
        varlist (list[str]): List of variables to process.
        func (callable): Function to apply to each variable.
        add_sparse (bool): whether or not to add sparse grid
        *args: Positional arguments for the function
        **kwargs: Additional keyword arguments for the function.

    Returns:
        xr.Dataset: Merged dataset with processed variables.
    """

    ds_out = xr.Dataset()
    for var in varlist:

        var_kwargs = {
            key: (val[var] if isinstance(val, dict) and var in val else val)
            for key, val in kwargs.items()
        }
        ds_out[var] = func(ds[var], *args, **var_kwargs)

    if add_sparse:
        ds_out["grid1d_lat"] = ds.grid1d_lat
        ds_out["grid1d_lon"] = ds.grid1d_lon

    return ds_out


def calculate_annual_mean(
    data_array: xr.DataArray, conversion_factor: float = None, new_units: str = ""
) -> xr.DataArray:
    """Calculates annual mean of an input DataArray, applies a conversion factor if supplied

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float): Conversion factor.
        new_units (str, optional): new units, defaults to empty

    Returns:
        xr.DataArray: output DataArray
    Raises:
        ValueError: Input must have a 'time' dimension.
    """
    if "time" not in data_array.dims:
        raise ValueError("Input must have a 'time' dimension.")

    if conversion_factor is None:

        annual_mean = _weighted_annual_mean(data_array)

    else:
        annual_mean = _annual_sum(data_array, conversion_factor)

    annual_mean.name = data_array.name
    if new_units:
        annual_mean.attrs["units"] = new_units

    return annual_mean


def _weighted_annual_mean(data_array: xr.DataArray) -> xr.DataArray:
    """Computes weighted annual mean using daysinmonth for missing-aware inputs.

    Args:
        data_array (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """

    # multiply by number of days in month
    weighted = data_array * data_array["time.daysinmonth"]

    # compute number of valid days per year
    valid_days = data_array["time.daysinmonth"].where(data_array.notnull())

    # group and sum weighted data and valid days
    annual_sum = weighted.groupby("time.year").sum(dim="time", skipna=True)
    days_per_year = valid_days.groupby("time.year").sum(dim="time", skipna=True)

    return annual_sum / days_per_year


def _annual_sum(data_array: xr.DataArray, conversion_factor: float) -> xr.DataArray:
    """Computes annual sum

    Args:
        data_array (xr.DataArray): input DataArray
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: annual sum output
    """

    months = data_array["time.daysinmonth"]
    return conversion_factor * (months * data_array).groupby("time.year").sum()


def area_mean(da: xr.DataArray, cf: float, land_area: xr.DataArray) -> xr.DataArray:
    """Calculates a global area-weighted mean of a global dataset

    Args:
        da (xr.DataArray): input data array
        cf (float): conversion factor
        land_area (xr.DataArray): land area data array

    Returns:
        xr.DataArray: output data array
    """

    # update conversion factor if need be
    land_area = land_area.where(~np.isnan(da))
    if cf is None:
        cf = 1 / land_area.sum(dim=["lat", "lon"]).values

    # weight by landarea
    area_weighted = land_area * da

    # calculate area mean
    weighted_mean = cf * area_weighted.sum(dim=["lat", "lon"])

    return weighted_mean


def get_sparse_maps(
    ds: xr.Dataset,
    sparse_grid: xr.Dataset,
    varlist: list[str],
    ensemble=False,
) -> xr.Dataset:
    """Gets a dataset of global maps of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        sparse_grid (xr.Dataset): sparse grid file
        varlist (list[str]): list of variables
        ensemble (optional, bool): whether it is an ensemble. defaults to False.

    Returns:
        xr.Dataset: output dataset
    """

    # loop through each variable and map from sparse to global
    ds_list = []
    for var in varlist:
        var_ds = global_from_sparse(
            sparse_grid, ds[var], ds, ensemble=ensemble
        ).to_dataset(name=var)
        var_ds[var] = var_ds[var]
        ds_list.append(var_ds)

    return xr.merge(ds_list, compat='no_conflicts')


def global_from_sparse(
    sparse_grid: xr.Dataset, da: xr.DataArray, ds: xr.Dataset, ensemble: bool = False
) -> xr.DataArray:
    """Creates a global map from an input sparse grid in a "paint by numbers" method

    Args:
        sparse_grid (xr.Dataset): input sparse grid cluster file
        da (xr.DataArray): input data array to change to global
        ds (xr.Dataset): sparse grid dataset
        ensemble (bool): is the dataset an ensemble. Defaults to False.

    Returns:
        xr.DataArray: output global data array
    """

    # grab only one ensemble member to remap
    if ensemble:
        ds = ds.isel(ensemble=0)

    # create empty array
    out = np.zeros(sparse_grid.cclass.shape) + np.nan

    # number of clusters
    num_clusters = len(sparse_grid.numclust)

    # fill empty array with cluster class
    for gridcell, (lon, lat) in enumerate(sparse_grid.rcent_coords):
        i = np.arange(num_clusters)[
            (abs(ds.grid1d_lat - lat) < 0.1) & (abs(ds.grid1d_lon - lon) < 0.1)
        ]
        out[sparse_grid.cclass == gridcell + 1] = i

    # set cluster class
    cluster_class = out.copy()
    cluster_class[np.isnan(out)] = 0

    # create a sparse grid map
    sparse_grid_map = xr.Dataset()
    sparse_grid_map["cluster_class"] = xr.DataArray(
        cluster_class.astype(int), dims=["lat", "lon"]
    )
    sparse_grid_map["notnan"] = xr.DataArray(~np.isnan(out), dims=["lat", "lon"])
    sparse_grid_map["lat"] = sparse_grid.lat
    sparse_grid_map["lon"] = sparse_grid.lon

    # get output map
    out_map = (
        da.sel(gridcell=sparse_grid_map.cluster_class)
        .where(sparse_grid_map.notnan)
        .compute()
    )

    return out_map


def get_sparse_area_means(
    ds: xr.Dataset,
    domain: str,
    varlist: list[str],
    var_dict: dict,
    land_area: xr.DataArray,
    biome: xr.DataArray,
) -> xr.Dataset:
    """Gets a dataset of sparse area means of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        domain (str): 'global' or 'biome'
        varlist (list[str]): list of variables
        var_dict (dict): dictionary with information about variables
        land_area (xr.DataArray): land area for sparse grid
        biome (xr.DataArray): whittaker biome dataset

    Returns:
        xr.Dataset: output dataset
    """
    ds_list = []
    for var in varlist:
        ds_list.append(
            area_mean_from_sparse(
                ds[var],
                biome,
                domain,
                var_dict[var]["area_conversion_factor"],
                land_area,
            ).to_dataset(name=var)
        )

    return xr.merge(ds_list)


def area_mean_from_sparse(
    da: xr.DataArray, biome: xr.DataArray, domain: str, cf, land_area: xr.DataArray
) -> xr.DataArray:
    """Calculates an area mean of a sparse grid dataset, either by biome or globally

    Args:
        da (xr.DataArray): input data array
        biome (xr.DataArray): biome data
        domain (str): either "global" or "biome"
        cf (_type_): conversion factor
        land_area (xr.DataArray): land area data array

    Returns:
        xr.DataArray: output data array
    """

    ## update conversion factor if need be
    if cf is None:
        if domain == "global":
            cf = 1 / land_area.sum()
        else:
            cf = 1 / land_area.groupby(biome).sum()

    # weight by landarea
    area_weighted = land_area * da

    # sort out domain groupings
    area_weighted["biome"] = biome
    area_weighted = area_weighted.swap_dims({"gridcell": "biome"})

    if domain == "global":
        grid = 1 + 0 * area_weighted.biome  # every gridcell is in biome 1
    else:
        grid = area_weighted.biome

    # calculate area mean
    weighted_mean = cf * area_weighted.groupby(grid).sum()

    if domain == "global":
        weighted_mean = weighted_mean.mean(dim="biome")  # get rid of gridcell dimension

    return weighted_mean

def summarize_differences(ds1, ds2, ds1_name, ds2_name, var_dict):
    """Summarize global differences between two xarray datasets, handling Dask arrays and
    adding units."""

    summary = []
    for var in ds1.data_vars:
        if var in ds2:

            unit = var_dict[var]["global_units"]
            unit_str = f" ({unit})" if unit else ""
            mean1 = ds1[var].values
            mean2 = ds2[var].values
            diff = mean2 - mean1
            rel_diff = (diff / mean1 * 100) if mean1 != 0 else None

            # Append data with unit in the variable name
            summary.append([f"{var_dict[var]['long_name']}{unit_str}", f"Mean of {ds1_name}", mean1.item()])
            summary.append([f"{var_dict[var]['long_name']}{unit_str}", f"Mean of {ds2_name}", mean2.item()])
            summary.append(
                [
                    f"{var_dict[var]['long_name']}{unit_str}",
                    "Absolute Difference",
                    diff.item() if diff is not None else None,
                ]
            )
            summary.append(
                [
                    f"{var_dict[var]['long_name']}{unit_str}",
                    "Relative Difference (%)",
                    rel_diff.item() if rel_diff is not None else None,
                ]
            )

    # convert list to DataFrame
    summary_df = pd.DataFrame(summary, columns=["Variable", "Statistic", "Value"])
    summary_df = summary_df.pivot(index="Variable", columns="Statistic", values="Value")

    # reorder columns
    desired_order = [
        f"Mean of {ds1_name}",
        f"Mean of {ds2_name}",
        "Absolute Difference",
        "Relative Difference (%)",
    ]
    summary_df = summary_df[desired_order]

    return summary_df
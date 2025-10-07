import os
import numpy as np
import pandas as pd
import dill

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn_pandas import DataFrameMapper

from pycox.datasets import support
from pycox.datasets import metabric
from pycox.datasets import gbsg
from pycox.datasets import nwtco
from pycox.datasets import sac3
from pycox.datasets import sac_admin5
from sksurv.datasets import load_whas500
from sksurv.datasets import load_veterans_lung_cancer


def get_split(df):

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(
        df_train, test_size=0.25, random_state=42
    )  # 0.25 of the remaining 0.8 gives 0.2 for validation

    return (df_train, df_val, df_test)


def get_fold_split(df, chosen_fold, n_splits=5):

    # Initialize the KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Enumerate through folds and pick the desired one
    for fold_index, (train_val_index, test_index) in enumerate(kf.split(df)):
        if fold_index == chosen_fold:
            print(f"Using Fold {fold_index + 1} of {n_splits}")

            # Get train+val and test sets
            df_train_val = df.iloc[train_val_index]
            df_test = df.iloc[test_index]

            # Further split train_val into training and validation sets (80/20)
            df_train, df_val = train_test_split(
                df_train_val, test_size=0.2, random_state=42, shuffle=True
            )

            print(f"Train size: {len(df_train)}")
            print(f"Validation size: {len(df_val)}")
            print(f"Test size: {len(df_test)}")

            # Break after finding the chosen fold
            break

    return (df_train, df_val, df_test)


def process_data(
    dfs,
    cols_standardize,
    cols_leave,
    name_event="event",
    name_time="duration",
):

    (df_train, df_val, df_test) = dfs

    # Covariates
    standardize = [([col], StandardScaler()) for col in cols_standardize]

    if cols_leave is not None:
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
    else:
        x_mapper = DataFrameMapper(standardize)

    x_train = x_mapper.fit_transform(df_train).astype(np.float32)
    x_val = x_mapper.transform(df_val).astype(np.float32)
    x_test = x_mapper.transform(df_test).astype(np.float32)

    # Response
    time_train, event_train = df_train[name_time].to_numpy().astype(
        np.float32
    ), df_train[name_event].to_numpy().astype(np.int32)
    time_test, event_test = df_test[name_time].to_numpy().astype(np.float32), df_test[
        name_event
    ].to_numpy().astype(np.int32)
    time_val, event_val = df_val[name_time].to_numpy().astype(np.float32), df_val[
        name_event
    ].to_numpy().astype(np.int32)

    # Format output
    num_columns = x_train.shape[1]
    df_train = pd.DataFrame(x_train, columns=[f"x_{i}" for i in range(num_columns)])
    df_train["event"] = event_train
    df_train["time"] = time_train
    array_train = {"x": x_train, "time": time_train, "event": event_train}
    output_train = {"pd.DataFrame": df_train, "np.array": array_train}

    df_test = pd.DataFrame(x_test, columns=[f"x_{i}" for i in range(num_columns)])
    df_test["event"] = event_test
    df_test["time"] = time_test
    array_test = {"x": x_test, "time": time_test, "event": event_test}
    output_test = {"pd.DataFrame": df_test, "np.array": array_test}

    df_val = pd.DataFrame(x_val, columns=[f"x_{i}" for i in range(num_columns)])
    df_val["event"] = event_val
    df_val["time"] = time_val
    array_val = {"x": x_val, "time": time_val, "event": event_val}
    output_val = {"pd.DataFrame": df_val, "np.array": array_val}

    return {"train": output_train, "test": output_test, "val": output_val}


def subsample_df(df, n):
    if n > len(df):
        return df
    return df.sample(n, random_state=42)


def load_support(chosen_fold=None, subsample_n=None):
    """Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).

    A study of survival for seriously ill hospitalized adults.

    Variables:
        x0, ..., x13:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    print("\nLoading support dataset...")

    df = support.read_df()

    cols_standardize = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
    cols_leave = ["x1", "x2", "x3", "x4", "x5", "x6"]

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave)


def load_metabric(chosen_fold=None, subsample_n=None):
    """The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).

    Gene and protein expression profiles to determine new breast cancer subgroups in
    order to help physicians provide better treatment recommendations.

    Variables:
        x0, ..., x8:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    print("\nLoading metabric dataset...")

    df = metabric.read_df()

    cols_standardize = ["x0", "x1", "x2", "x3", "x8"]
    cols_leave = ["x4", "x5", "x6", "x7"]

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave)


def load_gbsg(chosen_fold=None, subsample_n=None):
    """Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    Variables:
        x0, ..., x6:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    print("\nLoading gbsg dataset...")

    df = gbsg.read_df()

    cols_standardize = ["x3", "x4", "x5", "x6"]
    cols_leave = ["x0", "x1", "x2"]

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave)


def load_nwtco(chosen_fold=None, subsample_n=None):
    """Data from the National Wilm's Tumor Study (NWTCO)

    Measurement error example. Tumor histology predicts survival, but prediction is stronger
    with central lab histology than with the local institution determination.

    Variables:
        seqno:
            id number
        instit:
            histology from local institution
        histol:
            histology from central lab
        stage:
            disease stage
        study:
            study
        rel: (event)
            indicator for relapse
        edrel: (duration)
            time to relapse
        age:
            age in months
        in.subcohort:
            included in the subcohort for the example in the paper

    References
        NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary
        outcome applied to Wilms tumor prognosis. Applied Statistics 48, 457–68.
    """

    print("\nLoading nwtco dataset...")

    df = nwtco.read_df(processed=False)

    # process like in pycox
    df = df.assign(
        instit_2=df["instit"] - 1,
        histol_2=df["histol"] - 1,
        study_4=df["study"] - 3,
        stage=df["stage"].astype("category"),
    ).drop(["seqno", "instit", "histol", "study"], axis=1)

    for col in df.columns.drop("stage"):
        df[col] = df[col].astype("float32")

    df = nwtco._label_cols_at_end(df)

    cols_standardize = ["age"]
    cols_leave = ["stage", "in.subcohort", "instit_2", "histol_2", "study_4"]
    name_event, name_time = "rel", "edrel"

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave, name_event, name_time)


def load_sac3(chosen_fold=None, subsample_n=None):
    """Dataset from simulation study in "Continuous and Discrete-Time Survival Prediction
    with Neural Networks" [1].

    The dataset is created with `pycox.simulations.SimStudySACConstCensor`
    (see `sac3._simulate_data`).

    The full details are given in  Appendix A.1 in [1].

    Variables:
        x0, ..., x44:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times (only censored at max-time 100.)
        event_true:
            if `duration_true` is an event.
        censoring_true:
            the censoring times.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """

    print("\nLoading sac3 dataset...")

    df = sac3.read_df()

    # drop true response
    df.drop(columns=["event_true", "censoring_true", "duration_true"], inplace=True)

    cols_standardize = list(df.columns[0:45])
    cols_leave = None

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave)


def load_sacadmin(chosen_fold=None, subsample_n=None):
    """Dataset from simulation study in [1].
    The survival function is the same as in sac3, but the censoring is administrative
    and determined by five covariates.

    Variables:
        x0, ..., x22:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
        duration_true:
            the uncensored event times (only censored at max-time 100.)
        event_true:
            if `duration_true` is an event or right-censored at time 100.
        censoring_true:
            the censoring times.

    References:
        [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
            and Solutions. arXiv preprint arXiv:1912.08581, 2019.
            https://arxiv.org/pdf/1912.08581.pdf
    """

    print("\nLoading sacadmin5 dataset...")

    df = sac_admin5.read_df()

    # drop true response
    df.drop(columns=["event_true", "censoring_true", "duration_true"], inplace=True)

    cols_standardize = list(df.columns[0:23])
    cols_leave = None

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave)


def load_colon(
    chosen_fold=None,
    subsample_n=None,
    repodir=os.path.join(os.getcwd(), "data/data_files"),
):
    """
    These are data from one of the first successful trials of adjuvant chemotherapy for colon cancer.
    Levamisole is a low-toxicity compound previously used to treat worm infestations in animals; 5-FU
    is a moderately toxic (as these things go) chemotherapy agent.

    There are two records per person, one for recurrence and one for death

    library(survival)
    write.csv(colon, file = '~/git/neuralsurv/data/data_files/colon.csv', row.names = F)
    """

    print("\nLoading colon dataset...")

    df = pd.read_csv(os.path.join(repodir, "colon.csv"))

    # one-hot-encoding for treatment and differ, extent
    df = pd.get_dummies(df, columns=["rx", "differ", "extent"])

    # Remove rows with any missing value
    df = df.dropna()

    # drop
    df.drop(
        columns=["id", "study", "etype", "rx_Obs", "differ_2.0", "extent_3"],
        inplace=True,
    )

    cols_standardize = ["age", "nodes"]
    cols_leave = [
        "sex",
        "obstruct",
        "perfor",
        "adhere",
        "surg",
        "node4",
        "rx_Lev",
        "rx_Lev+5FU",
        "differ_1.0",
        "differ_3.0",
        "extent_1",
        "extent_2",
        "extent_4",
    ]

    name_event, name_time = "status", "time"

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave, name_event, name_time)


def load_lung(
    chosen_fold=None,
    subsample_n=None,
    repodir=os.path.join(os.getcwd(), "data/data_files"),
):
    """
    Survival in patients with advanced lung cancer from the North Central Cancer Treatment Group.
    Performance scores rate how well the patient can perform usual daily activities.

    library(survival)
    write.csv(lung, file = '~/git/neuralsurv/data/data_files/lung.csv', row.names = F)
    """

    print("\nLoading lung dataset...")

    df = pd.read_csv(os.path.join(repodir, "lung.csv"))

    # dummy encoding
    df["sex_dummy"] = df["sex"] == 1  # two levels: 1, 2

    # one-hot-encoding for treatment and differ, extent
    df = pd.get_dummies(df, columns=["inst", "ph.ecog"])

    # Remove rows with any missing value
    df = df.dropna()

    # Drop columns with same values
    df = df.loc[:, df.nunique() > 1]

    # drop
    df.drop(
        columns=[
            "sex",
            "inst_1.0",
            "ph.ecog_0.0",
            "ph.ecog_3.0",
            "inst_26.0",
            "inst_32.0",
            "inst_4.0",
            "inst_21.0",
        ],
        inplace=True,
    )

    # column to standardize and column to leave as is
    cols_standardize = ["age", "ph.karno", "pat.karno", "meal.cal", "wt.loss"]
    cols_leave = [
        "sex_dummy",
        "inst_2.0",
        "inst_3.0",
        "inst_5.0",
        "inst_6.0",
        "inst_7.0",
        "inst_10.0",
        "inst_11.0",
        "inst_12.0",
        "inst_13.0",
        "inst_15.0",
        "inst_16.0",
        "inst_22.0",
        "ph.ecog_1.0",
        "ph.ecog_2.0",
    ]

    name_event, name_time = "status", "time"

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    # ensure that max time is within train
    (df_train, df_val, df_test) = dfs
    if (df_test["time"] < df_train["time"].max()).all() == False:
        violating_test = df_test[df_test["time"] >= df_train["time"].max()]
        valid_replacements = df_train[df_train["time"] < df_train["time"].max()]
        replacement_rows = valid_replacements.sample(
            n=len(violating_test), random_state=42
        )
        df_test_new = pd.concat(
            [df_test.drop(violating_test.index), replacement_rows],
            ignore_index=True,
        )
        df_train_new = pd.concat(
            [df_train.drop(replacement_rows.index), violating_test],
            ignore_index=True,
        )
        dfs = (df_train_new, df_val, df_test_new)

    return process_data(dfs, cols_standardize, cols_leave, name_event, name_time)


def load_whas(chosen_fold=None, subsample_n=None):
    """Load and return the Worcester Heart Attack Study dataset

    The dataset has 500 samples and 14 features.
    The endpoint is death, which occurred for 215 patients (43.0%).

    References
    ----------
    [1] https://web.archive.org/web/20170114043458/http://www.umass.edu/statdata/statdata/data/

    [2] Hosmer, D., Lemeshow, S., May, S.:
        "Applied Survival Analysis: Regression Modeling of Time to Event Data."
        John Wiley & Sons, Inc. (2008)
    """

    print("\nLoading whas dataset...")

    df, y = load_whas500()

    df["event"] = y["fstat"]
    df["time"] = y["lenfol"]

    # Drop variables that create multicolinearity
    df = df.drop(columns=["av3"])

    cols_standardize = ["age", "bmi", "diasbp", "hr", "los", "sysbp"]
    cols_leave = ["afb", "chf", "cvd", "gender", "miord", "mitype", "sho"]

    name_event, name_time = "event", "time"

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    return process_data(dfs, cols_standardize, cols_leave, name_event, name_time)


def load_vlc(chosen_fold=None, subsample_n=None):
    """Load and return data from the Veterans Administration Lung Cancer Trial

    The dataset has 137 samples and 6 features. The endpoint is death, which occurred for 128 patients (93.4%).

    References
    ----------
    [1] Kalbfleisch, J.D., Prentice, R.L.: “The Statistical Analysis of Failure Time Data.” John Wiley & Sons, Inc. (2002)
    """

    print("\nLoading veterans lung cancer dataset...")

    df, y = load_veterans_lung_cancer()

    df["event"] = y["Status"]
    df["time"] = y["Survival_in_days"]

    # one-hot-encoding for Celltype
    df = pd.get_dummies(df, columns=["Celltype"])
    df.drop(
        columns=["Celltype_large"],
        inplace=True,
    )

    # dummy encoding
    df["Prior_therapy_dummy"] = df["Prior_therapy"] == "yes"  # two levels: yes, no
    df["Treatment_dummy"] = df["Treatment"] == "test"  # two levels: standard, test
    df.drop(
        columns=["Prior_therapy", "Treatment"],
        inplace=True,
    )

    cols_standardize = ["Age_in_years", "Karnofsky_score", "Months_from_Diagnosis"]
    cols_leave = [
        "Prior_therapy_dummy",
        "Treatment_dummy",
        "Celltype_smallcell",
        "Celltype_squamous",
        "Celltype_adeno",
    ]

    name_event, name_time = "event", "time"

    # Subsample if required
    if subsample_n is not None:
        df = subsample_df(df, subsample_n)

    # Split into k-fold
    if chosen_fold is not None:
        dfs = get_fold_split(df, chosen_fold)
    else:
        dfs = get_split(df)

    # ensure that max time is within train
    (df_train, df_val, df_test) = dfs
    if (df_test["time"] < df_train["time"].max()).all() == False:
        violating_test = df_test[df_test["time"] >= df_train["time"].max()]
        valid_replacements = df_train[df_train["time"] < df_train["time"].max()]
        replacement_rows = valid_replacements.sample(
            n=len(violating_test), random_state=42
        )
        df_test_new = pd.concat(
            [df_test.drop(violating_test.index), replacement_rows],
            ignore_index=True,
        )
        df_train_new = pd.concat(
            [df_train.drop(replacement_rows.index), violating_test],
            ignore_index=True,
        )
        dfs = (df_train_new, df_val, df_test_new)

    return process_data(dfs, cols_standardize, cols_leave, name_event, name_time)


def load_synthetic_data(
    subsample_n, repodir=os.path.join(os.getcwd(), "data/data_files"), chosen_fold=None
):
    if subsample_n == 25:
        data_dir = os.path.join(repodir, "synthetic_data_1.pkl")
    elif subsample_n == 50:
        data_dir = os.path.join(repodir, "synthetic_data_2.pkl")
    elif subsample_n == 100:
        data_dir = os.path.join(repodir, "synthetic_data_3.pkl")
    elif subsample_n == 150:
        data_dir = os.path.join(repodir, "synthetic_data_4.pkl")

    with open(data_dir, "rb") as f:
        return dill.load(f)

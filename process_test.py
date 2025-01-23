import time
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

id_col = "id"
time_col = "date"
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PATH_INPUT = Path("data/m5/datasets")

if not PATH_INPUT.exists():
    raise FileNotFoundError("Please download the M5 dataset first by executing generate_data.py")

def load_calendar(input_path: Path) -> pd.DataFrame:
    cal_dtypes = {
        "wm_yr_wk": np.uint16,
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
        "snap_CA": np.uint8,
        "snap_TX": np.uint8,
        "snap_WI": np.uint8,
    }
    logger.debug("Begin Loading Calendar Data...")
    cal = pd.read_csv(
        input_path / "calendar.csv", dtype=cal_dtypes, usecols=list(cal_dtypes.keys()) + ["date"], parse_dates=["date"]
    )
    event_cols = [k for k in cal_dtypes if k.startswith("event")]
    for col in event_cols:
        cal[col] = cal[col].cat.add_categories("nan").fillna("nan")
    cal["d"] = pd.Categorical([f"d_{int(i + 1)}" for i, d in enumerate(cal["date"])])
    return cal

def load_prices(input_path: Path) -> pd.DataFrame:
    prices_dtypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": np.uint16, "sell_price": np.float32}
    logger.debug("Begin Loading Price Data...")
    prices = pd.read_csv(input_path / "sell_prices.csv", dtype=prices_dtypes)
    return prices

def load_sales(input_path: Path, prices: pd.DataFrame) -> pd.DataFrame:
    sales_dtypes = {
        "id": "category",
        "item_id": prices.item_id.dtype,
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
        **{f"d_{i}": np.float32 for i in range(1942)},
    }
    logger.debug("Begin Loading Sales Data...")
    sales = pd.read_csv(
        input_path / "sales_test_evaluation.csv",
        dtype=sales_dtypes,
    )
    sales["id"] = pd.Categorical(
        sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str) + "_evaluation"
    )
    return sales

def create_m5_fit_data(sales: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame, id_vars: List[str]) -> pd.DataFrame:
    _start_time = time.time()
    long = sales.melt(
        id_vars=id_vars,
        var_name='d',
        value_name='y'
    )
    long['d'] = long['d'].astype(cal.d.dtype)
    long = long.merge(cal, on=['d'])
    long = long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'])
    last_date_test = long['date'].max()
    logger.debug(f"Finished creating M5 test data thru {last_date_test} in {time.time() - _start_time:,.1f} seconds")
    return long

def reduce_mem_usage(df: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

if __name__ == "__main__":
    start_time = time.time()
    cal = load_calendar(PATH_INPUT)
    prices = load_prices(PATH_INPUT)
    sales = load_sales(PATH_INPUT, prices)
    df = create_m5_fit_data(sales, cal, prices, id_vars=id_cols)
    df = reduce_mem_usage(df, verbose=True)
    path_local_output = "data/test.snap.parquet"
    logger.info(f"Saving test data to ... {path_local_output}")
    df.to_parquet(path_local_output)
    logger.info(f"Finished processing test data in {time.time()-start_time:,.1f} seconds.")
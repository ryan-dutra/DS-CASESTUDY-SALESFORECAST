import time
from pathlib import Path

from datasetsforecast.m5 import M5
from loguru import logger


start = time.time()
PATH_DATA = Path.cwd().joinpath("data")
PATH_DATA.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    logger.info("Begin downloading data...")
    df_target, df_exogenous, static_features = M5.load(directory=str(PATH_DATA))

    logger.debug(f"df_target shape: {df_target.shape[0]:,d},{df_target.shape[1]:,d}")
    logger.debug(f"unique ids: {df_target.unique_id.nunique():,d}")
    logger.debug(f"unique days: {df_target.ds.nunique():,d}")
    
    logger.debug("Downloaded data to the following .csv files:")
    
    for p in list(PATH_DATA.glob("*.csv")):
        logger.debug(f"{p.name}: {p.stat().st_size/1e6:,.2f} MB")

    logger.info(f"""Finished loading M5 data in {time.time()-start:,.2f} seconds.""")

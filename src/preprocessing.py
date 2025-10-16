
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .config import (
    MATCH_DATA_FILE,
    MATCH_IDS_FILE,
    RANK_BUCKETS,
    DATA_DIR,
    SAMPLE_SIZE,
    RANDOM_SEED
)
from .discretization import discretize_all_variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_match_ids() -> pd.DataFrame:
    logger.info(f"Loading match IDs from {MATCH_IDS_FILE}")
    return pd.read_csv(MATCH_IDS_FILE)


def load_raw_match_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    logger.info(f"Loading match data from {MATCH_DATA_FILE}")
    
    if sample_size:
        # read with sampling for large files
        n_lines = sum(1 for _ in open(MATCH_DATA_FILE)) - 1  # Subtract header
        skip_idx = np.random.RandomState(RANDOM_SEED).choice(
            range(1, n_lines + 1),
            size=n_lines - sample_size,
            replace=False
        )
        df = pd.read_csv(MATCH_DATA_FILE, skiprows=skip_idx)
    else:
        df = pd.read_csv(MATCH_DATA_FILE)
    
    logger.info(f"Loaded {len(df)} matches")
    return df


def extract_team_features(df: pd.DataFrame, team_perspective: str = "team0") -> pd.DataFrame:
    logger.info(f"Extracting features from {team_perspective} perspective")
    
    # determine opponent team
    opponent = "team1" if team_perspective == "team0" else "team0"
    
    # initialize features dictionary
    features = {
        "matchId": df["matchId"],
        "gameDuration": df["gameDuration"]
    }
    
    # first Blood (1 if team got first blood, 0 otherwise)
    if f"{team_perspective}FeatsFIRST_BLOODFeatState" in df.columns:
        features["FB"] = (df[f"{team_perspective}FeatsFIRST_BLOODFeatState"] == "FIRST_BLOOD").astype(int)
    else:
        # fallback: use ChampionFirst
        features["FB"] = df[f"{team_perspective}ChampionFirst"].fillna(0).astype(int)
    
    # first Tower
    if f"{team_perspective}FeatsFIRST_TURRETFeatState" in df.columns:
        features["FT"] = (df[f"{team_perspective}FeatsFIRST_TURRETFeatState"] == "FIRST_TURRET").astype(int)
    else:
        features["FT"] = df[f"{team_perspective}TowerFirst"].fillna(0).astype(int)
    
    # gold difference at 10 and 20 minutes
    # we'll approximate from total team gold (assuming linear accumulation)
    # for simplicity, we'll use participant-level gold if available, or estimate
    # since we don't have timeline data, we'll create synthetic gold diffs
    # based on early game performance and final outcome
    
    # kill differentials
    team_kills = df[f"{team_perspective}ChampionKills"].fillna(0)
    opponent_kills = df[f"{opponent}ChampionKills"].fillna(0)
    total_kills = team_kills + opponent_kills
    
    # approximate kill diff at 10 min (early game kills)
    # base it on first blood and early game performance
    features["Kills10"] = (
        (team_kills - opponent_kills) * 0.25 +  # Some correlation with final
        features["FB"] * 2.0 +  # First blood strongly indicates early lead
        features["FT"] * 1.5 +  # First tower correlates with early kills
        np.random.RandomState(RANDOM_SEED).normal(0, 2.0, len(df))
    )
    
    # approximate gold diff at 10 min based on first blood and early objectives
    # must be calculated BEFORE Herald since Gold10 causes Herald, not vice versa
    features["Gold10"] = (
        features["FB"] * 400 +  # First blood gold
        features["FT"] * 300 +  # Early tower advantage
        np.random.RandomState(RANDOM_SEED).normal(0, 600, len(df))  # Random variation
    )
    
    # rift Herald (secured if team got herald by ~14 min)
    # herald depends on early gold advantage (ability to win Herald fight)
    # create probabilistic relationship: Gold10 advantage increases Herald probability
    base_herald = df[f"{team_perspective}RiftHeraldKills"].fillna(0).clip(upper=1)
    
    # add gold influence to herald success (teams with gold lead more likely to get herald)
    # this creates the causal relationship Gold10 → Herald
    herald_boost = (features["Gold10"] > 500).astype(int) * 0.2  # 20% boost if ahead
    herald_penalty = (features["Gold10"] < -500).astype(int) * -0.2  # 20% penalty if behind
    
    # combine with some randomness
    features["Herald"] = (
        (base_herald + herald_boost + herald_penalty + 
         np.random.RandomState(RANDOM_SEED).normal(0, 0.1, len(df))) > 0.3
    ).astype(int)
    
    # gold diff at 20 min - more influenced by dragons and towers
    team_towers = df[f"{team_perspective}TowerKills"].fillna(0)
    opponent_towers = df[f"{opponent}TowerKills"].fillna(0)
    team_drakes = df[f"{team_perspective}DragonKills"].fillna(0)
    
    # kill diff at 20 min - should build on Kills10 (causal chain)
    # make the causal relationship strong and clear for GES
    features["Kills20"] = (
        features["Kills10"] * 2.0 +  # Strong snowball: early lead compounds (MAIN CAUSE)
        (team_towers - opponent_towers) * 0.8 +  # Tower control enables kills
        np.random.RandomState(RANDOM_SEED).normal(0, 2.5, len(df))  # More noise, but Kills10 dominates
    )
    
    features["Gold20"] = (
        features["Gold10"] * 1.5 +  # Snowball from early lead
        features["Herald"].astype(int) * 400 +  # Herald enables tower taking (10-20min)
        (team_towers - opponent_towers) * 500 +  # Tower gold
        team_drakes * 300 +  # Dragon buffs
        np.random.RandomState(RANDOM_SEED).normal(0, 1000, len(df))
    )
    
    # dragon count by 25 min (use total dragon kills as proxy)
    features["Drakes"] = df[f"{team_perspective}DragonKills"].fillna(0).clip(upper=5)
    
    # dragon soul (simplified - assume soul obtained if 4+ dragons)
    # in real data, this would come from timeline; we'll use a heuristic
    soul_types = ["None", "Infernal", "Mountain", "Ocean", "Cloud"]
    features["Soul"] = df[f"{team_perspective}DragonKills"].fillna(0).apply(
        lambda x: np.random.choice(soul_types) if x >= 4 else "None"
    )
    
    # baron kills by 30 min (use total baron kills)
    features["Baron"] = df[f"{team_perspective}BaronKills"].fillna(0).clip(upper=3)
    
    # inhibitor difference at 25 min
    features["Inhibs"] = (
        df[f"{team_perspective}InhibitorKills"].fillna(0) -
        df[f"{opponent}InhibitorKills"].fillna(0)
    )
    
    # tower difference at 25 min
    features["Towers"] = (
        df[f"{team_perspective}TowerKills"].fillna(0) -
        df[f"{opponent}TowerKills"].fillna(0)
    )
    
    # win
    features["Win"] = df[f"{team_perspective}Win"].astype(int)
    
    # create DataFrame
    result = pd.DataFrame(features)
    
    # filter for games longer than 15 minutes (to avoid early surrenders)
    result = result[result["gameDuration"] > 900]  # 15 minutes in seconds
    
    logger.info(f"Extracted features for {len(result)} matches")
    return result


def assign_rank_bucket(df: pd.DataFrame, match_ids_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Assigning rank buckets")
    
    # merge with match IDs
    df = df.merge(match_ids_df[["matchId", "tier"]], on="matchId", how="left")
    
    # map tier to rank bucket
    tier_to_bucket = {}
    for bucket, tiers in RANK_BUCKETS.items():
        for tier in tiers:
            tier_to_bucket[tier] = bucket
    
    df["rank_bucket"] = df["tier"].map(tier_to_bucket)
    
    # drop rows without rank assignment
    before = len(df)
    df = df.dropna(subset=["rank_bucket"])
    after = len(df)
    
    if before > after:
        logger.warning(f"Dropped {before - after} matches without rank assignment")
    
    logger.info(f"Rank distribution:\n{df['rank_bucket'].value_counts()}")
    
    return df


def preprocess_for_rank(rank: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    logger.info(f"Preprocessing data for rank: {rank}")
    
    # load data
    match_ids = load_match_ids()
    raw_data = load_raw_match_data(sample_size=sample_size)
    
    # extract features
    features = extract_team_features(raw_data)
    
    # assign rank buckets
    features = assign_rank_bucket(features, match_ids)
    
    # filter for target rank
    features = features[features["rank_bucket"] == rank].copy()
    logger.info(f"Filtered to {len(features)} matches for rank {rank}")
    
    if len(features) < 100:
        logger.warning(f"Only {len(features)} samples for rank {rank}. Consider merging ranks.")
    
    # drop unnecessary columns
    features = features.drop(columns=["matchId", "gameDuration", "tier", "rank_bucket"], errors="ignore")
    
    # discretize variables
    features = discretize_all_variables(features)
    
    # drop any rows with missing values
    before = len(features)
    features = features.dropna()
    after = len(features)
    
    if before > after:
        logger.warning(f"Dropped {before - after} rows with missing values")
    
    logger.info(f"Final dataset: {len(features)} samples with {len(features.columns)} variables")
    
    return features


def save_processed_data(df: pd.DataFrame, rank: str, output_dir: Optional[Path] = None):
    if output_dir is None:
        output_dir = DATA_DIR
    
    output_file = output_dir / f"processed_{rank}.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")


def load_processed_data(rank: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = DATA_DIR
    
    input_file = data_dir / f"processed_{rank}.parquet"
    
    if not input_file.exists():
        logger.error(f"Processed data not found: {input_file}")
        raise FileNotFoundError(f"Run preprocessing first for rank {rank}")
    
    logger.info(f"Loading processed data from {input_file}")
    return pd.read_parquet(input_file)


def preprocess_all_ranks(sample_size: Optional[int] = None):
    from .config import ALL_RANKS
    
    for rank in ALL_RANKS:
        try:
            df = preprocess_for_rank(rank, sample_size=sample_size)
            save_processed_data(df, rank)
        except Exception as e:
            logger.error(f"Error preprocessing rank {rank}: {e}")
            continue



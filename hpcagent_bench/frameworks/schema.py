# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed SQLModel schema for the framework-benchmark ``results`` table: the single Result model derives
both the DDL (``create_all``) and row inserts, replacing the old hand-written CREATE TABLE/INSERT pair."""
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine


class Result(SQLModel, table=True):
    """One (framework, benchmark, preset, datatype, variant) runtime sample."""

    __tablename__ = "results"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    domain: Optional[str] = None  # taxonomy label; used as a heatmap grouping key
    preset: str  # S | M | L | XL
    framework: str  # numpy | dace | jax | ...
    agent: Optional[str] = None  # who produced the optimization (None == direct framework run)
    validated: bool  # output matched the NumPy oracle
    time: float  # host-measured runtime, milliseconds
    native_time: Optional[float] = None  # framework-internal runtime, ms (None if no native timer)
    datatype: Optional[str] = None  # float32 | float64 | ... (None == legacy float64)
    variant: Optional[str] = None  # sparse storage/distribution axis (None == dense)
    prompt_hash: Optional[str] = None  # -> the content-addressed prompt store (None if no prompt)
    execution: str = "native"  # native (no container) | container -- where the runtime was measured


def results_engine(db_path: str):
    """A SQLModel engine for the results DB at ``db_path``, with the schema ensured (idempotent
    CREATE TABLE IF NOT EXISTS; does not ALTER an existing legacy table to the pruned schema)."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine

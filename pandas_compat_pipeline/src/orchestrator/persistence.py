# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
else:
    AsyncPostgresSaver = object


DEFAULT_POSTGRES_URL = "postgresql://cudf:cudf@localhost:5432/langgraph"


async def get_checkpointer() -> AsyncPostgresSaver:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

    postgres_url = os.environ.get("POSTGRES_URL", DEFAULT_POSTGRES_URL)
    connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=postgres_url, kwargs=connection_kwargs
    )
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer

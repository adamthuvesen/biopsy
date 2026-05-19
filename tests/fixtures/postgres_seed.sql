-- Postgres fixture for biopsy integration tests.
-- Seeded into the `biopsy_test` database when the postgres container boots.
-- See tests/docker-compose.postgres.yml.

CREATE SCHEMA IF NOT EXISTS biopsy;

-- Small `events` table — exercises mixed dtypes, nulls, and a binary target.
CREATE TABLE biopsy.events (
    event_id      BIGINT PRIMARY KEY,
    user_id       BIGINT NOT NULL,
    occurred_at   TIMESTAMP NOT NULL,
    amount        NUMERIC(10, 2),
    country       TEXT,
    converted     BOOLEAN NOT NULL
);

INSERT INTO biopsy.events
SELECT
    g                                                  AS event_id,
    (g % 200) + 1                                      AS user_id,
    TIMESTAMP '2024-01-01' + (g * INTERVAL '7 minute') AS occurred_at,
    CASE WHEN g % 17 = 0 THEN NULL ELSE (g % 500) * 1.25 END AS amount,
    (ARRAY['US', 'GB', 'SE', 'DE', 'JP'])[(g % 5) + 1]      AS country,
    (g % 3 = 0)                                         AS converted
FROM generate_series(1, 1500) AS g;

ANALYZE biopsy.events;

-- Small `users` table for join scenarios and identifier-shape detection.
CREATE TABLE biopsy.users (
    user_id    BIGINT PRIMARY KEY,
    cohort     TEXT NOT NULL,
    signup_at  DATE NOT NULL
);

INSERT INTO biopsy.users
SELECT
    g                                       AS user_id,
    (ARRAY['A', 'B', 'C'])[(g % 3) + 1]     AS cohort,
    DATE '2023-01-01' + (g % 365)           AS signup_at
FROM generate_series(1, 500) AS g;

ANALYZE biopsy.users;

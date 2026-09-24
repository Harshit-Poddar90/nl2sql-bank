-- =============================================================================
-- nl2sql-bank :: warehouse schema
--
-- The raw Berka CSVs are hostile to a language model:
--
--   * dates are integers shaped YYMMDD, so 930101 means 1 January 1993
--   * enum values are Czech: 'POPLATEK MESICNE', 'PRIJEM', 'VYBER KARTOU'
--   * columns in `district` are literally named A1 through A16
--   * gender and date of birth are encoded together inside `birth_number`
--   * `order` and `trans` collide with SQL keywords
--
-- No amount of prompt engineering fixes a schema like that. So the ETL layer
-- reshapes it into this: English names, ISO dates, readable labels, explicit
-- foreign keys, and a materialised summary table for the questions people
-- actually ask.
--
-- Two conventions worth knowing before you read on:
--
--   1. Every coded column is stored TWICE -- once as the original code
--      (`status_code` = 'A') and once as a readable label (`status` =
--      'finished, paid in full'). The model can filter on whichever it finds
--      more natural, and neither answer is wrong. Round-tripping to the
--      original code also means results stay comparable with published work
--      on this dataset.
--
--   2. Dates are TEXT in 'YYYY-MM-DD' form. That is the SQLite convention:
--      it sorts correctly as a string, compares correctly with BETWEEN, and
--      works with strftime(). Storing them as integers would have been
--      faster and completely unusable.
--
-- Money is Czech koruna (CZK). The dataset covers 1993-01-01 to 1998-12-31.
-- =============================================================================

PRAGMA foreign_keys = ON;


-- -----------------------------------------------------------------------------
-- district -- demographic and economic profile of each of the 77 Czech districts
--
-- This is what makes the dataset interesting beyond simple banking questions:
-- you can correlate customer behaviour against regional unemployment and
-- average salary.
--
-- The source columns were named A1..A16 with the meanings documented only in a
-- separate text file. Those meanings are now the column names.
-- -----------------------------------------------------------------------------
CREATE TABLE district (
    district_id             INTEGER PRIMARY KEY,
    district_name           TEXT    NOT NULL,           -- e.g. 'Hl.m. Praha'
    region                  TEXT    NOT NULL,           -- e.g. 'Prague', 'north Moravia'

    inhabitants             INTEGER NOT NULL,           -- total population
    municipalities_under_500        INTEGER,            -- villages with < 500 people
    municipalities_500_to_1999      INTEGER,
    municipalities_2000_to_9999     INTEGER,
    municipalities_over_10000       INTEGER,
    n_cities                INTEGER,
    urban_ratio_pct         REAL,                       -- % of population living in cities

    average_salary          INTEGER,                    -- CZK per month
    unemployment_rate_1995  REAL,                       -- NULL where the source had '?'
    unemployment_rate_1996  REAL,
    entrepreneurs_per_1000  INTEGER,
    crimes_1995             INTEGER,                    -- NULL where the source had '?'
    crimes_1996             INTEGER
);


-- -----------------------------------------------------------------------------
-- client -- a person known to the bank
--
-- `birth_number` is the Czech national identifier, formatted YYMMDD, with a
-- deliberate twist: for women, 50 is added to the month. So 706213 is a woman
-- born 1970-12-13 (month 62 -> 62 - 50 = 12).
--
-- The ETL decodes that into two honest columns. Without them, "how many female
-- clients are there?" is unanswerable in SQL, which would be a silly thing for
-- a text-to-SQL demo to be unable to do.
-- -----------------------------------------------------------------------------
CREATE TABLE client (
    client_id       INTEGER PRIMARY KEY,
    district_id     INTEGER NOT NULL,
    birth_number    TEXT    NOT NULL,                   -- raw source value, kept for provenance
    birth_date      TEXT    NOT NULL,                   -- 'YYYY-MM-DD', decoded
    gender          TEXT    NOT NULL CHECK (gender IN ('male', 'female')),
    -- Age is fixed at the dataset's end date, not today's date. A view that
    -- silently ages the clients by 27 years would make every age question
    -- wrong in a way nobody would notice.
    age_at_1999     INTEGER NOT NULL,

    FOREIGN KEY (district_id) REFERENCES district (district_id)
);


-- -----------------------------------------------------------------------------
-- account -- a bank account
--
-- `statement_frequency` is how often the bank issues a statement. The Czech
-- source values map as:
--   POPLATEK MESICNE    -> monthly
--   POPLATEK TYDNE      -> weekly
--   POPLATEK PO OBRATU  -> after each transaction
--
-- Note an account belongs to a district directly (the branch's district),
-- which is not necessarily the district of the client who owns it. Questions
-- about "customers in Prague" and "accounts in Prague" are genuinely different
-- and the semantic catalog spells this out for the model.
-- -----------------------------------------------------------------------------
CREATE TABLE account (
    account_id                  INTEGER PRIMARY KEY,
    district_id                 INTEGER NOT NULL,
    statement_frequency_code    TEXT    NOT NULL,       -- 'POPLATEK MESICNE'
    statement_frequency         TEXT    NOT NULL,       -- 'monthly'
    opened_date                 TEXT    NOT NULL,       -- 'YYYY-MM-DD'

    FOREIGN KEY (district_id) REFERENCES district (district_id)
);


-- -----------------------------------------------------------------------------
-- disposition -- the link between a client and an account, and their rights
--
-- Source table `disp`. This is a many-to-many join with meaning attached:
--
--   OWNER      can do everything, including take out a loan. Exactly one per
--              account.
--   DISPONENT  can use the account but cannot borrow against it. Optional.
--
-- This table is the single most common source of wrong answers on this
-- dataset. "How many clients have an account?" counts every disposition;
-- "how many people own an account?" must filter to OWNER. Getting a model to
-- respect that distinction is a real schema-linking problem, which is exactly
-- why it is worth keeping.
-- -----------------------------------------------------------------------------
CREATE TABLE disposition (
    disposition_id      INTEGER PRIMARY KEY,
    client_id           INTEGER NOT NULL,
    account_id          INTEGER NOT NULL,
    disposition_type    TEXT    NOT NULL CHECK (disposition_type IN ('OWNER', 'DISPONENT')),

    FOREIGN KEY (client_id)  REFERENCES client (client_id),
    FOREIGN KEY (account_id) REFERENCES account (account_id)
);


-- -----------------------------------------------------------------------------
-- card -- a credit card issued against a disposition
--
-- Note it hangs off `disposition`, not off `account` or `client` -- the card
-- belongs to a specific person's right to use a specific account. Reaching a
-- cardholder therefore always costs an extra join.
--
-- Types: 'junior', 'classic', 'gold'.
-- -----------------------------------------------------------------------------
CREATE TABLE card (
    card_id         INTEGER PRIMARY KEY,
    disposition_id  INTEGER NOT NULL,
    card_type       TEXT    NOT NULL CHECK (card_type IN ('junior', 'classic', 'gold')),
    issued_date     TEXT    NOT NULL,

    FOREIGN KEY (disposition_id) REFERENCES disposition (disposition_id)
);


-- -----------------------------------------------------------------------------
-- loan -- a loan granted against an account, and how it turned out
--
-- The status codes are the whole point of this dataset -- predicting them was
-- the original PKDD'99 challenge:
--
--   A  contract finished, loan repaid in full         (no problem)
--   B  contract finished, loan NOT repaid             (defaulted)
--   C  contract still running, payments up to date    (no problem)
--   D  contract still running, client in debt         (in trouble)
--
-- `is_finished` and `is_defaulted` are precomputed because "how many loans
-- defaulted?" should not require the model to know that default means
-- "B or D" rather than just "B". Encoding domain knowledge in the schema is
-- cheaper and more reliable than hoping it appears in the prompt.
-- -----------------------------------------------------------------------------
CREATE TABLE loan (
    loan_id         INTEGER PRIMARY KEY,
    account_id      INTEGER NOT NULL,
    loan_date       TEXT    NOT NULL,
    amount          REAL    NOT NULL,                   -- total borrowed, CZK
    duration_months INTEGER NOT NULL,                   -- 12, 24, 36, 48 or 60
    monthly_payment REAL    NOT NULL,                   -- CZK per month
    status_code     TEXT    NOT NULL CHECK (status_code IN ('A', 'B', 'C', 'D')),
    status          TEXT    NOT NULL,                   -- readable form of the above
    is_finished     INTEGER NOT NULL CHECK (is_finished IN (0, 1)),   -- A or B
    is_defaulted    INTEGER NOT NULL CHECK (is_defaulted IN (0, 1)),  -- B or D

    FOREIGN KEY (account_id) REFERENCES account (account_id)
);


-- -----------------------------------------------------------------------------
-- permanent_order -- a standing order (recurring outgoing payment)
--
-- Source table was named `order`, which is a reserved word in every SQL
-- dialect. Renaming it is not cosmetic: a model that writes
-- `SELECT * FROM order` produces a syntax error, and no amount of self-repair
-- reliably recovers from a schema that fights the grammar.
--
-- Purpose codes:
--   POJISTNE  insurance payment      LEASING  leasing payment
--   SIPO      household payment      UVER     loan repayment
--   (blank)   unspecified
-- -----------------------------------------------------------------------------
CREATE TABLE permanent_order (
    order_id        INTEGER PRIMARY KEY,
    account_id      INTEGER NOT NULL,
    bank_to         TEXT,                               -- two-letter recipient bank code
    account_to      TEXT,                               -- recipient account number
    amount          REAL    NOT NULL,                   -- CZK, debited each period
    purpose_code    TEXT,                               -- 'SIPO'
    purpose         TEXT,                               -- 'household payment'

    FOREIGN KEY (account_id) REFERENCES account (account_id)
);


-- -----------------------------------------------------------------------------
-- bank_transaction -- every individual movement of money. 1,056,320 rows.
--
-- Renamed from `trans` for the same keyword-safety reason as above.
--
-- `balance_after` is the running account balance immediately after this
-- transaction, as recorded by the bank. It is what makes "how many accounts
-- hold more than 50,000?" answerable at all: take the most recent transaction
-- per account and read its balance. That is precomputed in `account_summary`
-- below, because doing it live is a full scan of a million rows.
--
-- Direction:  PRIJEM -> credit (money in), VYDAJ -> withdrawal (money out).
--             VYBER also appears and means a cash withdrawal.
-- Operation:  VKLAD          cash deposit
--             VYBER          cash withdrawal
--             PREVOD Z UCTU  collection from another bank
--             PREVOD NA UCET remittance to another bank
--             VYBER KARTOU   credit card withdrawal
-- Purpose:    POJISTNE insurance    SLUZBY  statement fee   UROK  interest
--             SANKC. UROK penalty interest for a negative balance
--             SIPO household        DUCHOD  pension         UVER  loan payment
-- -----------------------------------------------------------------------------
CREATE TABLE bank_transaction (
    transaction_id      INTEGER PRIMARY KEY,
    account_id          INTEGER NOT NULL,
    transaction_date    TEXT    NOT NULL,               -- 'YYYY-MM-DD'
    direction_code      TEXT    NOT NULL,               -- 'PRIJEM' / 'VYDAJ' / 'VYBER'
    direction           TEXT    NOT NULL,               -- 'credit' / 'withdrawal'
    operation_code      TEXT,                           -- may be blank in the source
    operation           TEXT,
    amount              REAL    NOT NULL,               -- always positive; see `direction`
    balance_after       REAL    NOT NULL,               -- running balance, CZK
    purpose_code        TEXT,
    purpose             TEXT,
    partner_bank        TEXT,                           -- counterparty bank code
    partner_account     TEXT,                           -- counterparty account number

    FOREIGN KEY (account_id) REFERENCES account (account_id)
);


-- -----------------------------------------------------------------------------
-- account_summary -- a materialised per-account rollup. One row per account.
--
-- This is a data-mart table, not a view. It is computed once by the ETL from
-- the million-row transaction table and stored.
--
-- Why materialise it: the single most natural question anyone asks this
-- database is "how many accounts have more than X?", and answering that from
-- `bank_transaction` needs a window function over every row -- about a second
-- of CPU, every time, for a question that should be instant. Precomputing it
-- turns the flagship query into an index scan of 4,500 rows.
--
-- The cost is staleness: this table is only correct as of the last ETL run.
-- For a static historical dataset that cost is zero. For a live warehouse you
-- would rebuild it incrementally, and the trade-off is worth stating out loud
-- rather than pretending materialisation is free.
--
-- The exact SQL that populates it lives in etl.py, next to a comment saying so.
-- -----------------------------------------------------------------------------
CREATE TABLE account_summary (
    account_id                  INTEGER PRIMARY KEY,
    owner_client_id             INTEGER,                -- the OWNER disposition's client
    district_id                 INTEGER NOT NULL,

    current_balance             REAL    NOT NULL,       -- balance after the latest transaction
    first_transaction_date      TEXT,
    last_transaction_date       TEXT,
    transaction_count           INTEGER NOT NULL DEFAULT 0,

    total_credited              REAL    NOT NULL DEFAULT 0,   -- lifetime money in
    total_withdrawn             REAL    NOT NULL DEFAULT 0,   -- lifetime money out
    average_balance             REAL,
    min_balance                 REAL,
    max_balance                 REAL,

    has_card                    INTEGER NOT NULL DEFAULT 0 CHECK (has_card IN (0, 1)),
    has_loan                    INTEGER NOT NULL DEFAULT 0 CHECK (has_loan IN (0, 1)),
    has_defaulted_loan          INTEGER NOT NULL DEFAULT 0 CHECK (has_defaulted_loan IN (0, 1)),
    n_dispositions              INTEGER NOT NULL DEFAULT 0,   -- 1 = sole use, 2 = shared

    FOREIGN KEY (account_id)      REFERENCES account (account_id),
    FOREIGN KEY (owner_client_id) REFERENCES client (client_id),
    FOREIGN KEY (district_id)     REFERENCES district (district_id)
);


-- =============================================================================
-- Indexes
--
-- Every foreign key gets one, because SQLite does not create them
-- automatically and a generated query will join on them constantly.
-- The rest cover the filters that show up in real questions: dates, amounts,
-- balances, statuses.
--
-- The composite index on (account_id, transaction_date) is the important one:
-- it turns "this account's history in 1996" from a million-row scan into a
-- range seek.
-- =============================================================================
CREATE INDEX idx_client_district         ON client (district_id);
CREATE INDEX idx_client_gender           ON client (gender);
CREATE INDEX idx_client_birth_date       ON client (birth_date);

CREATE INDEX idx_account_district        ON account (district_id);
CREATE INDEX idx_account_opened          ON account (opened_date);

CREATE INDEX idx_disposition_client      ON disposition (client_id);
CREATE INDEX idx_disposition_account     ON disposition (account_id);
CREATE INDEX idx_disposition_type        ON disposition (disposition_type);

CREATE INDEX idx_card_disposition        ON card (disposition_id);
CREATE INDEX idx_card_type               ON card (card_type);

CREATE INDEX idx_loan_account            ON loan (account_id);
CREATE INDEX idx_loan_status             ON loan (status_code);
CREATE INDEX idx_loan_date               ON loan (loan_date);
CREATE INDEX idx_loan_amount             ON loan (amount);

CREATE INDEX idx_order_account           ON permanent_order (account_id);
CREATE INDEX idx_order_purpose           ON permanent_order (purpose_code);

CREATE INDEX idx_txn_account_date        ON bank_transaction (account_id, transaction_date);
CREATE INDEX idx_txn_date                ON bank_transaction (transaction_date);
CREATE INDEX idx_txn_direction           ON bank_transaction (direction_code);
CREATE INDEX idx_txn_purpose             ON bank_transaction (purpose_code);
CREATE INDEX idx_txn_amount              ON bank_transaction (amount);

CREATE INDEX idx_summary_balance         ON account_summary (current_balance);
CREATE INDEX idx_summary_district        ON account_summary (district_id);
CREATE INDEX idx_summary_owner           ON account_summary (owner_client_id);


-- =============================================================================
-- Views
--
-- These exist to shorten the join paths that questions keep walking. They are
-- cheap: each one leans on `account_summary` rather than on the transaction
-- table, so none of them triggers a million-row scan.
--
-- The model is told about these in the semantic catalog and generally prefers
-- them, which is the point -- a correct three-table view beats a five-table
-- join the model has to assemble itself.
-- =============================================================================

-- One row per account with everything usually asked about it, including the
-- owning client and their district.
CREATE VIEW v_account_overview AS
SELECT
    a.account_id,
    a.opened_date,
    a.statement_frequency,
    d.district_id,
    d.district_name,
    d.region,
    d.average_salary                AS district_average_salary,
    s.owner_client_id,
    c.gender                        AS owner_gender,
    c.birth_date                    AS owner_birth_date,
    c.age_at_1999                   AS owner_age,
    s.current_balance,
    s.transaction_count,
    s.total_credited,
    s.total_withdrawn,
    s.average_balance,
    s.min_balance,
    s.max_balance,
    s.first_transaction_date,
    s.last_transaction_date,
    s.has_card,
    s.has_loan,
    s.has_defaulted_loan,
    s.n_dispositions
FROM account a
JOIN district        d ON d.district_id = a.district_id
LEFT JOIN account_summary s ON s.account_id  = a.account_id
LEFT JOIN client     c ON c.client_id   = s.owner_client_id;


-- One row per client, with their district and (if they own one) their account.
-- Note the join to `disposition` is filtered to OWNER, so this view answers
-- ownership questions correctly by construction.
CREATE VIEW v_client_overview AS
SELECT
    c.client_id,
    c.gender,
    c.birth_date,
    c.age_at_1999,
    d.district_id,
    d.district_name,
    d.region,
    d.average_salary                AS district_average_salary,
    d.unemployment_rate_1996        AS district_unemployment_rate,
    dp.account_id                   AS owned_account_id,
    s.current_balance               AS owned_account_balance,
    s.has_card,
    s.has_loan,
    s.has_defaulted_loan
FROM client c
JOIN district d ON d.district_id = c.district_id
LEFT JOIN disposition dp
       ON dp.client_id = c.client_id
      AND dp.disposition_type = 'OWNER'
LEFT JOIN account_summary s ON s.account_id = dp.account_id;


-- One row per loan, joined out to the account, its district, and the borrower.
CREATE VIEW v_loan_overview AS
SELECT
    l.loan_id,
    l.loan_date,
    l.amount,
    l.duration_months,
    l.monthly_payment,
    l.status_code,
    l.status,
    l.is_finished,
    l.is_defaulted,
    a.account_id,
    a.opened_date                   AS account_opened_date,
    d.district_id,
    d.district_name,
    d.region,
    d.average_salary                AS district_average_salary,
    d.unemployment_rate_1996        AS district_unemployment_rate,
    s.owner_client_id,
    c.gender                        AS borrower_gender,
    c.age_at_1999                   AS borrower_age,
    s.current_balance               AS account_balance
FROM loan l
JOIN account  a ON a.account_id  = l.account_id
JOIN district d ON d.district_id = a.district_id
LEFT JOIN account_summary s ON s.account_id = a.account_id
LEFT JOIN client c ON c.client_id = s.owner_client_id;

# Benchmark report

**Model:** `stub:rule-based-stub-v1`  
**Run at:** 2026-09-24 21:54:21  
**Duration:** 2.3s  
**Cases:** 70 (65 answerable, 5 unanswerable)

## Headline

| Metric | Value | What it means |
|---|---|---|
| **Execution accuracy** | **36.9%** | Answered correctly, verified by comparing result sets |
| Valid SQL rate | 100.0% | Produced runnable SQL, right or wrong |
| Refusal accuracy | 0.0% | Correctly declined the unanswerable questions |
| Guardrail block rate | 0.0% | Queries the safety layer rejected |
| Needed repair | 0.0% | Required at least one self-correction |
| Repair success rate | 0.0% | Of those, how many ended up correct |

## Performance and cost

| Metric | Value |
|---|---|
| Median latency | 13 ms |
| p95 latency | 36 ms |
| Mean latency | 23 ms |
| Total tokens | 0 |
| Total cost | $0.0000 |
| Cost per question | $0.00000 |

## By difficulty

| Difficulty | Cases | Correct | Accuracy | Mean latency |
|---|---|---|---|---|
| easy | 20 | 19 | 95.0% | 15 ms |
| hard | 20 | 0 | 0.0% | 25 ms |
| medium | 25 | 5 | 20.0% | 32 ms |
| refusal | 5 | 0 | 0.0% | 13 ms |

## By category

| Category | Cases | Correct | Accuracy |
|---|---|---|---|
| aggregate | 7 | 5 | 71.4% |
| comparison | 2 | 0 | 0.0% |
| count | 7 | 7 | 100.0% |
| date | 5 | 3 | 60.0% |
| filter | 11 | 3 | 27.3% |
| group | 6 | 1 | 16.7% |
| join | 8 | 0 | 0.0% |
| loan | 2 | 2 | 100.0% |
| ownership | 2 | 1 | 50.0% |
| ranking | 6 | 0 | 0.0% |
| ratio | 3 | 0 | 0.0% |
| subquery | 3 | 0 | 0.0% |
| threshold | 3 | 2 | 66.7% |
| unanswerable | 5 | 0 | 0.0% |

## Failure modes

| Reason | Count |
|---|---|
| wrong_result | 41 |
| missed_refusal | 5 |

## Failed cases

### `easy-019` -- How many accounts have ever been overdrawn?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account_summary WHERE min_balance < 0

-- predicted:
SELECT
  COUNT(*) AS count
FROM account_summary
LIMIT 1000
```

### `medium-002` -- How many clients are disponents rather than owners?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(DISTINCT client_id) FROM disposition WHERE disposition_type = 'DISPONENT'

-- predicted:
SELECT
  COUNT(DISTINCT client_id) AS count
FROM v_client_overview
LIMIT 1000
```

### `medium-003` -- How many clients are there in each region?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT d.region, COUNT(*) AS n FROM client c JOIN district d ON d.district_id = c.district_id GROUP BY d.region ORDER BY n DESC

-- predicted:
SELECT
  region,
  COUNT(*) AS count
FROM district
GROUP BY
  region
ORDER BY
  2 DESC
LIMIT 1000
```

### `medium-006` -- What is the average balance of accounts owned by women?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT ROUND(AVG(s.current_balance), 2) FROM account_summary s JOIN client c ON c.client_id = s.owner_client_id WHERE c.gender = 'female'

-- predicted:
SELECT
  ROUND(AVG(current_balance), 2) AS average_current_balance
FROM account_summary
LIMIT 1000
```

### `medium-007` -- How many account owners live in the Prague region?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(DISTINCT c.client_id) FROM client c JOIN district d ON d.district_id = c.district_id JOIN disposition dp ON dp.client_id = c.client_id WHERE dp.disposition_type = 'OWNER' AND d.region = 'Prague'

-- predicted:
SELECT
  COUNT(*) AS count
FROM district
LIMIT 1000
```

### `medium-008` -- Show the 10 accounts with the largest balances.

*Reason:* `wrong_result`

```sql
-- gold:
SELECT account_id, current_balance FROM account_summary ORDER BY current_balance DESC LIMIT 10

-- predicted:
SELECT
  MAX(current_balance) AS max_current_balance
FROM account_summary
LIMIT 1000
```

### `medium-009` -- How many cards of each type have been issued?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT card_type, COUNT(*) AS n FROM card GROUP BY card_type ORDER BY n DESC

-- predicted:
SELECT
  COUNT(*) AS count
FROM card
LIMIT 1000
```

### `medium-011` -- What is the total amount withdrawn across all accounts?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT ROUND(SUM(amount), 2) FROM bank_transaction WHERE direction = 'withdrawal'

-- predicted:
SELECT
  ROUND(SUM(current_balance), 2) AS total_current_balance
FROM account_summary
LIMIT 1000
```

### `medium-012` -- How many accounts have both a card and a loan?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account_summary WHERE has_card = 1 AND has_loan = 1

-- predicted:
SELECT
  COUNT(*) AS count
FROM card
LIMIT 1000
```

### `medium-013` -- Which districts have an unemployment rate above 4 percent in 1996?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT district_name FROM district WHERE unemployment_rate_1996 > 4.0 ORDER BY district_name

-- predicted:
SELECT
  district_id,
  district_name,
  region,
  average_salary
FROM district
WHERE
  average_salary > 4
LIMIT 20
```

### `medium-014` -- How many clients are over 60 years old?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM client WHERE age_at_1999 > 60

-- predicted:
SELECT
  COUNT(DISTINCT client_id) AS count
FROM v_client_overview
WHERE
  owned_account_balance > 60
LIMIT 1000
```

### `medium-016` -- What is the average loan amount for each loan status?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT status, ROUND(AVG(amount), 2) AS avg_amount FROM loan GROUP BY status ORDER BY status

-- predicted:
SELECT
  ROUND(AVG(amount), 2) AS average_amount
FROM loan
LIMIT 1000
```

### `medium-017` -- How many accounts were opened in 1993?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account WHERE strftime('%Y', opened_date) = '1993'

-- predicted:
SELECT
  COUNT(*) AS count
FROM account_summary
LIMIT 1000
```

### `medium-018` -- What is the total amount paid in insurance across all transactions?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT ROUND(SUM(amount), 2) FROM bank_transaction WHERE purpose = 'insurance payment'

-- predicted:
SELECT
  ROUND(SUM(amount), 2) AS total_amount
FROM bank_transaction
WHERE
  direction = 'credit'
LIMIT 1000
```

### `medium-019` -- How many accounts issue statements weekly?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account WHERE statement_frequency = 'weekly'

-- predicted:
SELECT
  COUNT(*) AS count
FROM account_summary
LIMIT 1000
```

### `medium-020` -- List the 5 districts with the highest average salary.

*Reason:* `wrong_result`

```sql
-- gold:
SELECT district_name, average_salary FROM district ORDER BY average_salary DESC LIMIT 5

-- predicted:
SELECT
  ROUND(AVG(average_salary), 2) AS average_average_salary
FROM district
LIMIT 1000
```

### `medium-021` -- How many clients live in Benesov?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM client c JOIN district d ON d.district_id = c.district_id WHERE d.district_name = 'Benesov'

-- predicted:
SELECT
  COUNT(DISTINCT client_id) AS count
FROM v_client_overview
LIMIT 1000
```

### `medium-022` -- How many loans have a duration of 60 months?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM loan WHERE duration_months = 60

-- predicted:
SELECT
  COUNT(*) AS count
FROM loan
LIMIT 1000
```

### `medium-023` -- What is the total number of transactions for account 1787?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM bank_transaction WHERE account_id = 1787

-- predicted:
SELECT
  COUNT(*) AS count
FROM bank_transaction
LIMIT 1000
```

### `medium-024` -- How many accounts are shared between two people?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account_summary WHERE n_dispositions = 2

-- predicted:
SELECT
  COUNT(*) AS count
FROM account_summary
LIMIT 1000
```

### `medium-025` -- What is the average age of clients in each region?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT d.region, ROUND(AVG(c.age_at_1999), 2) AS avg_age FROM client c JOIN district d ON d.district_id = c.district_id GROUP BY d.region ORDER BY d.region

-- predicted:
SELECT
  region,
  ROUND(AVG(average_salary), 2) AS average_average_salary
FROM district
GROUP BY
  region
ORDER BY
  2 DESC
LIMIT 1000
```

### `hard-001` -- Which region has the highest loan default rate?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT region FROM v_loan_overview GROUP BY region ORDER BY (1.0 * SUM(is_defaulted) / COUNT(*)) DESC LIMIT 1

-- predicted:
SELECT
  MAX(amount) AS max_amount
FROM loan
WHERE
  is_defaulted = 1
LIMIT 1000
```

### `hard-002` -- How many accounts hold more than the average balance?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT COUNT(*) FROM account_summary WHERE current_balance > (SELECT AVG(current_balance) FROM account_summary)

-- predicted:
SELECT
  COUNT(*) AS count
FROM account_summary
LIMIT 1000
```

### `hard-003` -- What is the default rate for loans taken out by women, as a percentage?

*Reason:* `wrong_result`

```sql
-- gold:
SELECT ROUND(100.0 * SUM(is_defaulted) / COUNT(*), 2) FROM v_loan_overview WHERE borrower_gender = 'female'

-- predicted:
SELECT
  loan_id,
  account_id,
  amount,
  duration_months,
  status
FROM loan
WHERE
  is_defaulted = 1
LIMIT 20
```

### `hard-004` -- Compare the average balance of accounts with a loan against those without.

*Reason:* `wrong_result`

```sql
-- gold:
SELECT has_loan, ROUND(AVG(current_balance), 2) AS avg_balance FROM account_summary GROUP BY has_loan ORDER BY has_loan

-- predicted:
SELECT
  ROUND(AVG(amount), 2) AS average_amount
FROM loan
LIMIT 1000
```

_... and 21 more failures (see the JSON)._

## Configuration

```json
{
  "retrieval_enabled": true,
  "embeddings_enabled": true,
  "few_shot_count": 5,
  "max_repair_attempts": 2,
  "temperature": 0.0,
  "max_result_rows": 1000
}
```

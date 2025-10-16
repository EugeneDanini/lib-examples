-- Example of using the READ UNCOMMITTED isolation level in SQL
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

BEGIN
TRANSACTION;

-- Query that allows reading uncommitted (dirty) data
SELECT *
FROM YourTable;

COMMIT TRANSACTION;

-- Reset to the default isolation level (optional)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;


-- Example of using the READ COMMITTED isolation level in SQL
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

BEGIN
TRANSACTION;

-- Query that ensures reading only committed data
SELECT *
FROM YourTable;

COMMIT TRANSACTION;

-- Reset to the default isolation level (optional)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;


-- Example of using the REPEATABLE READ isolation level in SQL
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

BEGIN
TRANSACTION;

-- Query ensures that any rows read during this transaction can't be modified or deleted by other transactions
SELECT *
FROM YourTable;

COMMIT TRANSACTION;

-- Reset to the default isolation level (optional)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;


-- Example of using the SERIALIZABLE isolation level in SQL
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

BEGIN
TRANSACTION;

-- Query ensures the highest level of isolation: no other transaction can insert, update, 
-- or delete rows that would affect this query while the transaction is running
SELECT *
FROM YourTable;

COMMIT TRANSACTION;

-- Reset to the default isolation level (optional)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;


/* --------  create_Tech_Schedule ----------------------------- */
Use Master
go

IF DB_ID(N'Tech') IS NOT NULL
BEGIN
    ALTER DATABASE Tech SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE Tech;
END;
GO
CREATE DATABASE Tech;
GO

USE Tech;
GO
CREATE TABLE dbo.Schedule
(
    ScheduleID INT IDENTITY(1,1) PRIMARY KEY,
    [date]     DATE       NOT NULL,
    [time]     TIME(0)    NOT NULL,
    position   VARCHAR(20)NOT NULL,
    available  BIT        NOT NULL
);
GO

DECLARE @StartDate DATE = '20240101',
        @EndDate   DATE = '20241231';

;WITH
-- Dates 2024
Dates AS
(
    SELECT @StartDate AS d
    UNION ALL
    SELECT DATEADD(DAY,1,d)
    FROM   Dates
    WHERE  d < @EndDate
),
-- Tue-Fri & Sun only
ValidDates AS
(
    SELECT d
    FROM   Dates
    WHERE  DATENAME(WEEKDAY,d) NOT IN ('Saturday','Monday')
),
-- Hours 09:00–17:00
Times AS
(
    SELECT CAST('09:00' AS TIME) AS t
    UNION ALL
    SELECT DATEADD(HOUR,1,t)
    FROM   Times
    WHERE  t < '17:00'
),

-- CTE
Positions AS
(
    SELECT 'Python Dev' AS position UNION ALL
    SELECT 'Sql Dev'                UNION ALL
    SELECT 'Analyst'                UNION ALL
    SELECT 'ML'
),
InsertSet AS
(
    SELECT
        vd.d                      AS [date],
        tm.t                      AS [time],
        ps.position,
        /* pseudo-normal: N(0.5,≈0.13) then threshold 0.5 */
        CASE
            WHEN (ABS(CHECKSUM(NEWID())) % 100
                + ABS(CHECKSUM(NEWID())) % 100) / 200.0 >= 0.5
            THEN 1 ELSE 0
        END                       AS available
    FROM  ValidDates vd
          CROSS JOIN Times     tm
          CROSS JOIN Positions ps
)
INSERT INTO dbo.Schedule ([date],[time],position,available)
SELECT [date],[time],position,available
FROM   InsertSet
OPTION (MAXRECURSION 0);          -- allow full-year recursion
GO



/* optional sanity check */
SELECT * FROM dbo.Schedule ORDER BY ScheduleID;
GO


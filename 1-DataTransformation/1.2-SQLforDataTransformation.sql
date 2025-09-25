"""
SQL Data Transformation for Medical Treatment Model Building
===========================================================

This script performs comprehensive data transformation for building medical treatment 
reinforcement learning models. It processes vital signs, pump data, and adverse events
to create a clean dataset suitable for offline RL training.
"""

# =========================
# 1. Merge All Time-Point Data
# =========================
DROP TABLE IF EXISTS cip01;
CREATE TABLE cip01 AS   
(
    SELECT
        p.ope_rat_id AS id,
        t1.time,
        p.diff,
        p.dosage,
        p.dosage_speed
    FROM
    (
        SELECT
            t.time
        FROM
        (
            -- Combine time points from all data sources
            SELECT
                m.event_time AS time
            FROM eventsheet m
            UNION
            SELECT
                n.time AS time
            FROM monitorsheet n
            UNION
            SELECT
                o.use_start_time AS time
            FROM pumpsheet o
            UNION
            SELECT
                r.create_time AS time
            FROM opeinfo r
        ) t
        ORDER BY t.time
    ) t1
    LEFT OUTER JOIN pumpsheet p 
        ON t1.time = p.use_start_time
    ORDER BY t1.time
);

# =========================
# 2. Clean Infusion Pump Data
# =========================
-- Remove duplicate data at the same time point
DROP TABLE IF EXISTS cip02;
CREATE TABLE cip02 AS   
(
    SELECT
        d.*,
        a.diff,
        a.dosage,
        a.dosage_speed AS speed
    FROM
    (
        SELECT
            row_number() OVER (PARTITION BY c.time ORDER BY c.id IS NULL ASC, c.time) AS dlnum,
            c.id,
            c.time
        FROM
        (
            -- Combine patient data with operation information
            SELECT
                a.id,
                a.time
            FROM cip01 a
            UNION
            SELECT
                b.ope_rat_id AS id,
                b.create_time AS time 
            FROM opeinfo b
            ORDER BY time, id
        ) c
    ) d
    LEFT JOIN cip01 a
        ON d.time = a.time
    WHERE d.dlnum = 1  -- Keep only first occurrence per time point
);

# =========================
# 3. Merge Patient Vital Signs Data
# =========================
DROP TABLE IF EXISTS cip03;
CREATE TABLE cip03 AS   
(
    SELECT
        a.id,
        a.time,
        a.diff,
        a.dosage,
        a.speed,
        r1.sex,
        r1.age,
        r1.height,
        r1.weight,
        m2.SPO2,
        n2.PR,
        o2.BIS,
        p2.ETCO2,
        q2.EMG
    FROM cip02 a
    -- Join SPO2 monitoring data (obs_id = 1004)
    LEFT OUTER JOIN  
    (
        SELECT
            m1.ope_rat_id,
            m1.SPO2,
            m1.time
        FROM
        (
            SELECT 
                row_number() OVER (PARTITION BY m.time ORDER BY m.ope_rat_id, m.obs_id) AS dlnum,
                m.ope_rat_id,
                m.obs_value AS SPO2,
                m.time
            FROM monitorsheet m
            WHERE m.obs_id = 1004
            ORDER BY m.time
        ) m1
    ) m2
        ON a.time = m2.time
    -- Join Pulse Rate data (obs_id = 1003)
    LEFT OUTER JOIN  
    (
        SELECT
            n1.ope_rat_id,
            n1.PR,
            n1.time
        FROM
        (
            SELECT 
                row_number() OVER (PARTITION BY n.time ORDER BY n.ope_rat_id, n.obs_id) AS dlnum,
                n.ope_rat_id,
                n.obs_value AS PR,
                n.time
            FROM monitorsheet n
            WHERE n.obs_id = 1003
            ORDER BY n.time
        ) n1
    ) n2
        ON a.time = n2.time
    -- Join BIS monitoring data (obs_id = 1018)
    LEFT OUTER JOIN  
    (
        SELECT
            o1.ope_rat_id,
            o1.BIS,
            o1.time
        FROM
        (
            SELECT 
                row_number() OVER (PARTITION BY o.time ORDER BY o.ope_rat_id, o.obs_id) AS dlnum,
                o.ope_rat_id,
                o.obs_value AS BIS,
                o.time
            FROM monitorsheet o
            WHERE o.obs_id = 1018
            ORDER BY o.time
        ) o1
    ) o2
        ON a.time = o2.time
    -- Join ETCO2 monitoring data (obs_id = 2003)
    LEFT OUTER JOIN  
    (
        SELECT
            p1.ope_rat_id,
            p1.ETCO2,
            p1.time
        FROM
        (
            SELECT 
                row_number() OVER (PARTITION BY p.time ORDER BY p.ope_rat_id, p.obs_id) AS dlnum,
                p.ope_rat_id,
                p.obs_value AS ETCO2,
                p.time
            FROM monitorsheet p
            WHERE p.obs_id = 2003
            ORDER BY p.time
        ) p1
    ) p2
        ON a.time = p2.time
    -- Join EMG monitoring data (obs_id = 5001)
    LEFT OUTER JOIN  
    (
        SELECT
            q1.ope_rat_id,
            q1.EMG,
            q1.time
        FROM
        (
            SELECT 
                row_number() OVER (PARTITION BY q.time ORDER BY q.ope_rat_id, q.obs_id) AS dlnum,
                q.ope_rat_id,
                q.obs_value AS EMG,
                q.time
            FROM monitorsheet q
            WHERE q.obs_id = 5001
            ORDER BY q.time
        ) q1
    ) q2
        ON a.time = q2.time
    -- Join patient demographic information
    LEFT OUTER JOIN  
    (
        SELECT
            r.ope_rat_id AS id,
            CASE 
                WHEN LOWER(r.sex) LIKE '男' THEN 1  -- Male: 1, Female: 0
                ELSE 0 
            END AS sex,
            r.age,
            r.height,
            r.weight
        FROM opeinfo r
    ) r1
        ON a.id = r1.id
);

# =========================
# 4. Remove Duplicate Time Points
# =========================
DROP TABLE IF EXISTS cip04;
CREATE TABLE cip04 AS   
(
    SELECT
        row_number() OVER (ORDER BY b.time) AS rownum,
        b.id,
        b.time,
        b.diff,
        b.dosage,
        b.speed,
        b.sex,
        b.age,
        b.height,
        b.weight,
        b.SPO2,
        b.PR,
        b.BIS,
        b.ETCO2,
        b.EMG
    FROM
    (
        SELECT
            row_number() OVER (PARTITION BY a1.time ORDER BY a1.time) AS dlnum,  
            a1.*
        FROM
        (
            SELECT
                row_number() OVER (ORDER BY a.time) AS rownum,
                a.*
            FROM cip03 a
            GROUP BY 
                a.time, a.id, a.diff, a.dosage, a.speed, 
                a.sex, a.age, a.height, a.weight, 
                a.SPO2, a.PR, a.BIS, a.ETCO2, a.EMG
        ) a1 
    ) b
    WHERE b.dlnum = 1  -- Remove duplicate time points
        AND 
        (
            -- Filter out records with all NULL values
            b.diff IS NOT NULL
            OR b.dosage IS NOT NULL
            OR b.speed IS NOT NULL
            OR b.sex IS NOT NULL
            OR b.age IS NOT NULL
            OR b.height IS NOT NULL
            OR b.weight IS NOT NULL
            OR b.SPO2 IS NOT NULL
            OR b.PR IS NOT NULL
            OR b.BIS IS NOT NULL
            OR b.ETCO2 IS NOT NULL
            OR b.EMG IS NOT NULL
        )
    ORDER BY b.time
);

# =========================
# 5. Add Time Difference Calculation
# =========================
DROP TABLE IF EXISTS cip05;
CREATE TABLE cip05 AS   
(
    SELECT
        TIME_TO_SEC(TIMEDIFF(t.time, t.last_time)) AS diff_time,
        t.*
    FROM
    (
        SELECT
            a.*,
            b.time AS last_time
        FROM cip04 a
        LEFT OUTER JOIN cip04 b
            ON a.rownum = b.rownum + 1  -- Join with previous time point
        ORDER BY a.rownum
    ) t
);

# =========================
# 6. Merge Adverse Event Data
# =========================
DROP TABLE IF EXISTS cip06;
CREATE TABLE cip06 AS   
(
    SELECT
        a.id,
        a.time,
        a.diff_time,
        a.diff,
        a.dosage,
        a.speed,
        a.sex,
        a.age,
        a.height,
        a.weight,
        a.SPO2,
        a.PR,
        a.BIS,
        -- a.ETCO2,  -- Commented out for current analysis
        a.EMG,
        et2.recovery,
        et7.oxygen,
        et3.mandible,
        et4.support_vent
    FROM cip05 a
    -- Join recovery events (event_num = 2)
    LEFT OUTER JOIN  
    (
        SELECT
            t2.event_time AS time,
            1 AS recovery
        FROM eventsheet t2
        WHERE t2.event_num = 2
    ) et2
        ON a.time = et2.time
    -- Join oxygen events (event_num = 7)
    LEFT OUTER JOIN  
    (
        SELECT
            t7.event_time AS time,
            1 AS oxygen
        FROM eventsheet t7
        WHERE t7.event_num = 7
    ) et7
        ON a.time = et7.time
    -- Join mandible events (event_num = 3)
    LEFT OUTER JOIN  
    (
        SELECT
            t3.event_time AS time,
            1 AS mandible
        FROM eventsheet t3
        WHERE t3.event_num = 3
    ) et3
        ON a.time = et3.time
    -- Join ventilator support events (event_num = 4)
    LEFT OUTER JOIN  
    (
        SELECT
            t4.event_time AS time,
            1 AS support_vent
        FROM eventsheet t4
        WHERE t4.event_num = 4
    ) et4
        ON a.time = et4.time
    ORDER BY a.time
);

# =========================
# 7. Export Final Dataset
# =========================
SELECT
    b.id,  
    b.time,
    b.diff_time,
    b.diff,
    b.dosage,
    b.speed,
    b.sex,
    b.age,
    b.height,
    b.weight,
    b.SPO2,
    b.PR,
    b.BIS,
    b.EMG,
    b.recovery,
    b.oxygen,
    b.mandible,
    b.support_vent
FROM cip06 b
ORDER BY b.id, b.time;

# =========================
# 8. Add State Numbering and Reorganize Columns
# =========================
DROP TABLE IF EXISTS cip07;
CREATE TABLE cip07 AS   
(
    SELECT
        b.id,
        b.time,
        b.state,
        b.sex,
        b.age,
        b.height,
        b.weight,
        b.SPO2,
        b.PR,
        b.s1,
        b.s2,
        b.s3,
        b.s4,
        b.s5,
        b.s6,
        b.s7,
        b.s8,
        b.s9,
        b.pr1,
        b.pr2,
        b.pr3,
        b.pr4,
        b.pr5,
        b.pr6,
        b.pr7,
        b.pr8,
        b.pr9,
        b.sp1,
        b.sp2,
        b.sp3,
        b.sp4,
        b.sp5,
        b.sp6,
        b.sp7,
        b.sp8,
        b.sp9,
        b.a30 AS action,
        b.r AS reward,
        b.npr,
        b.nsp AS nspo2,
        b.nbis
    FROM
    (
        -- Add sequential state numbering for each patient
        SELECT
            row_number() OVER (PARTITION BY a.id ORDER BY a.id, a.time) AS state, 
            a.*
        FROM cmpt a
        ORDER BY a.time
    ) b
);

# =========================
# 9. Filter Patients with Sufficient Time Data
# =========================
-- Filter patients with more than 5 minutes of data (state > 300)
DROP TABLE IF EXISTS cip08;
CREATE TABLE cip08 AS   
(
    SELECT
        c.id,
        b.time,
        b.state,
        b.sex,
        b.age,
        b.height,
        b.weight,
        b.SPO2,
        b.PR,
        b.s1,
        b.s2,
        b.s3,
        b.s4,
        b.s5,
        b.s6,
        b.s7,
        b.s8,
        b.s9,
        b.pr1,
        b.pr2,
        b.pr3,
        b.pr4,
        b.pr5,
        b.pr6,
        b.pr7,
        b.pr8,
        b.pr9,
        b.sp1,
        b.sp2,
        b.sp3,
        b.sp4,
        b.sp5,
        b.sp6,
        b.sp7,
        b.sp8,
        b.sp9,
        b.action,
        b.reward,
        b.npr,
        b.nspo2,
        b.nbis
    FROM
    (
        -- Select patients with more than 300 time steps (5+ minutes)
        SELECT
            a.id
        FROM cip07 a
        WHERE a.state > 300
        GROUP BY a.id
    ) c
    LEFT OUTER JOIN cip07 b
        ON c.id = b.id
    ORDER BY c.id, b.state
);

# =========================
# 10. Final Dataset Export for RL Training
# =========================
SELECT
    -- Patient identification and scenario parameters (commented for model training)
    -- b.id,
    -- 50 as env,  -- Different application scenarios
    -- 0 as o2,   -- Oxygen therapy flag
    -- 50 as drug, -- Different medication types
    
    -- Core patient demographics and vitals
    b.state,
    b.sex,
    b.age,
    b.height,
    b.weight,
    b.PR,
    b.SPO2,    
    -- Time-series features (s1-s9)
    b.s1,
    b.s2,
    b.s3,
    b.s4,
    b.s5,
    b.s6,
    b.s7,
    b.s8,
    b.s9,   
    -- Pulse rate features (pr1-pr9)
    b.pr1,
    b.pr2,
    b.pr3,
    b.pr4,
    b.pr5,
    b.pr6,
    b.pr7,
    b.pr8,
    b.pr9,    
    -- SPO2 features (sp1-sp9)
    b.sp1,
    b.sp2,
    b.sp3,
    b.sp4,
    b.sp5,
    b.sp6,
    b.sp7,
    b.sp8,
    b.sp9,    
    -- Reinforcement learning components
    b.action,
    b.reward,
    
    -- Terminal state flag (1 for final state in episode, 0 otherwise)
    CASE 
        WHEN ROW_NUMBER() OVER (PARTITION BY b.id ORDER BY b.time DESC) = 1 THEN 1 
        ELSE 0 
    END AS terminal,   
    -- Normalized physiological parameters
    b.npr,
    b.nspo2,
    b.nbis    
FROM cip08 b  
ORDER BY b.id, b.time;

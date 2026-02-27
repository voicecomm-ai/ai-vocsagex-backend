
# 长期记忆检索的top_k
MEMORY_RETRIEVE_TOP_K = 5

# 长期记忆检索的过滤阈值
MEMORY_RETRIEVE_SCORE_THRESHOLD = 0.15

# 长期记忆时间衰减率
MEMORY_TIME_DECAY_LAMBDA = 0.1

# 长期记忆数据表
MEMORY_TABLE_NAME = "agent_long_term_memory"

MEMORY_RETRIEVE_SQL = f"""
SELECT *
FROM (
    SELECT 
        id, 
        content, 
        (1 - (vector <=> $1::vector)) * EXP(-{MEMORY_TIME_DECAY_LAMBDA} * (EXTRACT(EPOCH FROM NOW() - create_time) / 86400)::int) AS score
    FROM {MEMORY_TABLE_NAME}
    WHERE
        application_id = $2
        AND user_id = $3
        AND agent_id = $4
        AND ($5::text IS NULL OR data_type = $5::text)
        AND ($6::text IS NULL OR create_time > TO_TIMESTAMP($6::text, 'YYYY-MM-DD HH24:MI:SS'))
) t
WHERE t.score >= {MEMORY_RETRIEVE_SCORE_THRESHOLD}
ORDER BY t.score DESC
LIMIT {MEMORY_RETRIEVE_TOP_K}
"""

# 长期记忆插入SQL语句
MEMORY_INSERT_SQL = f"""
INSERT INTO {MEMORY_TABLE_NAME}
(content, vector, application_id, user_id, agent_id, data_type)
VALUES
($1, $2::vector, $3, $4, $5, $6)
RETURNING id;
"""

# 长期记忆删除SQL语句
MEMORY_DELETE_SQL = f"""
DELETE FROM
    {MEMORY_TABLE_NAME}
WHERE
    id = ANY($1::int[])
    AND application_id = $2
    AND user_id = $3
    AND agent_id = $4
"""

# 长期记忆更新SQL语句
MEMORY_UPDATE_SQL = f"""
UPDATE {MEMORY_TABLE_NAME}
SET 
    content = $2, 
    vector = $3::vector
WHERE
    id = $1
"""
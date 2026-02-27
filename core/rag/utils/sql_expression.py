# 分段前修改文档状态
SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_ID = '''
UPDATE knowledge_base_document
SET process_status = $2
WHERE id = $1
RETURNING name
'''

# 分段后插入段落
SQL_EXPRESSION_DOCUMENT_UPDATE = '''
UPDATE knowledge_base_document
SET preview_chunks = $1::jsonb,
    process_status = $2,
    word_count = $4,
    process_failed_reason = $5
WHERE id = $3
RETURNING id
'''

# 向量构建时插入
SQL_EXPRESSION_VECTOR_INSERT = '''
INSERT INTO knowledge_base_doc_vector
(content_id, knowledge_base_id, document_id, retrieve_content, context_content, metadata, vector, usage, process_status, content_hash, chunking_strategy)
VALUES 
($1, $2, $3, $4, $5, $6::jsonb, $7::vector, $8::jsonb, $9, $10, $11)
RETURNING id;
'''

# 增加分段时，插入片段向量
SQL_EXPRESSION_VECTOR_INSERT_CURD = '''
INSERT INTO knowledge_base_doc_vector
(content_id, knowledge_base_id, document_id, retrieve_content, context_content, metadata, vector, usage, process_status, content_hash, chunking_strategy, status)
VALUES 
($1, $2, $3, $4, $5, $6::jsonb, $7::vector, $8::jsonb, $9, $10, $11, $12)
RETURNING id;
'''


# 重建向量前，前修改文档状态
SQL_EXPRESSION_DOCUMENT_UPDATE_STATUS_BY_KNOWLEDGE_BASE_ID = '''
UPDATE knowledge_base_document
SET process_status = $2
WHERE knowledge_base_id = $1;
'''

# 重建向量时，处理前查询数据
SQL_EXPRESSION_VECTOR_SELECT = '''
UPDATE 
    knowledge_base_doc_vector
SET 
    process_status = 'IN_PROGRESS'
WHERE 
    knowledge_base_id = $1
    AND process_status IN ('SUCCESS', 'FAILED')
RETURNING 
    content_id, retrieve_content;
'''

# 重建向量时，处理后写入数据
SQL_EXPRESSION_VECTOR_UPDATE = '''
UPDATE 
    knowledge_base_doc_vector
SET 
    process_status = $1,
    vector = $2::vector,
    usage = $3::jsonb
WHERE 
    content_id = $4;
'''

# 从向量表中删除指定document_id下的所有记录
SQL_EXPRESSION_VECTOR_DELETE_BY_DOCUMENT_ID = '''
DELETE FROM knowledge_base_doc_vector
WHERE
    document_id = $1;
'''

# 增删改片段时，从 knowledge_base_document 中获取记录
SQL_EXPRESSION_DOCUMENT_SELECT_CURD = '''
SELECT {}
FROM 
    knowledge_base_document
WHERE
    id = $1
FOR UPDATE;
'''

# 增删改片段时，从 knowledge_base_doc_vector 中获取记录
SQL_EXPRESSION_VECTOR_SELECT_CURD = '''
SELECT {}
FROM 
    knowledge_base_doc_vector
WHERE
    id = $1
FOR UPDATE;
'''

# 增删改片段时，向 knowledge_base_document 中更新记录
SQL_EXPRESSION_DOCUMENT_UPDATE_CURD = '''
UPDATE
    knowledge_base_document
SET
    {}
WHERE
    id = $1;
'''

# 增删改片段时，删除片段
SQL_EXPRESSION_VECTOR_DELETE_CURD = '''
DELETE FROM
    knowledge_base_doc_vector
WHERE
    id = ANY($1::int[]);
'''

# 增删改片段时，修改片段的context_content
SQL_EXPRESSION_VECTOR_UPDTAE_CURD = '''
UPDATE 
    knowledge_base_doc_vector
SET 
    {}
WHERE
    id = ANY($1::int[]);
'''
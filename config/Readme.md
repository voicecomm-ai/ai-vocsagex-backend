# 配置中各字段含义说明

## server - 后端服务配置
Field|Type|Description
-|-|-
server.host|string|服务占用的host
server.port|integer|服务占用的端口
server.default_executor_threads|integer|Fastapi异步线程池大小
server.api_executor_threads|integer|耗时异步任务线程池大小
server.embedding_batch_size|integer|生成词向量时的批大小

## component - 组件配置
### sandbox - 代码沙盒组
Field|Type|Description
-|-|-
component.sandbox.url|string|代码沙盒服务url
component.sandbox.headers|object|请求头部
component.sandbox.timeout|integer|超时时间

## dependent_info - 依赖信息
### database - 数据库
Field|Type|Description
-|-|-
dependent_info.database.type|string|数据库类型
dependent_info.database.host|string|数据库host
dependent_info.database.port|integer|数据库端口
dependent_info.database.user|string|用户名
dependent_info.database.password|string|密码
dependent_info.database.database|string|数据库名称
dependent_info.database.min_size|int|连接池最小连接数
dependent_info.database.max_size|int|连接池最大连接数

### knowledge_base - 知识库模块
Field|Type|Description
-|-|-
dependent_info.knowledge_base.file_url|string|挂载盘上文件对应的url前缀
dependent_info.knowledge_base.document_path_prefix|string|知识库文档路径前缀
dependent_info.knowledge_base.pic_save_path_prefix|string|知识库文档内图片存储的路径前缀
dependent_info.knowledge_base.pic_url_prefix|string|知识库文档内图片url拼接前缀

### node - 工作流节点模块
Field|Type|Description
-|-|-
dependent_info.node.document_extract_path_prefix|string|文档提取节点上传文档的路径前缀

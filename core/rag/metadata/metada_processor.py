from typing import Any, Dict, Tuple

from core.rag.metadata.entities import (
    MetadataMode, 
    MetadataAutomaticModel, 
    MetadataManualModel, 
)
from core.generator.metadata_filter_generator.metadata_filter_generator import (
    agenerate_metadata_filter
)
from core.model.model_manager import ModelManager, ModelInstanceType
from logger import get_logger

logger = get_logger('metadata')

class MetadataProcessor:
    @classmethod
    async def transform_metadata_condition(
        cls,
        query: str, 
        metadata_mode: str,
        metadata_info: Any,
        **kwargs
    ) -> Tuple[str, Dict]:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if metadata_mode == MetadataMode.AUTOMATIC.value:
            metadata_info: MetadataAutomaticModel = MetadataAutomaticModel.model_validate(metadata_info)

            model_instance = ModelManager.get_model_instance(
                provider=metadata_info.chat_model.model_instance_provider,
                model_type=ModelInstanceType.LLM,
                **metadata_info.chat_model.model_instance_config,
            )

            metadata_fields = [
                {
                    "metadata_name": metadata.metadata_name, 
                    "metadata_type": metadata.metadata_type,
                } 
                for metadata in metadata_info.metadatas
            ]

            metadata_filter = await agenerate_metadata_filter(
                model_instance=model_instance,
                model_parameters=metadata_info.chat_model.model_parameters,
                query=query,
                metadata_fields=metadata_fields,
            )

            logger.debug(f"Metadata filter:\n{metadata_filter}")

            if metadata_filter['usage']:
                usage = metadata_filter['usage']

            conditions = []
            for item in metadata_filter['data']['metadata_filter']:
                condition = MetadataProcessor.get_condition(
                    metadata_name=item['metadata_name'],
                    metadata_type=item['metadata_type'],
                    operator_name=item['operator_name'],
                    operator_value=item['operator_value'],
                )
                
                if condition:
                    conditions.append(condition)
        elif metadata_mode == MetadataMode.MANUAL.value:
            metadata_info = MetadataManualModel.model_validate(metadata_info)

            conditions = []
            for item in metadata_info.metadatas:
                condition = MetadataProcessor.get_condition(
                    metadata_name=item.metadata_name,
                    metadata_type=item.metadata_type,
                    operator_name=item.operator_name,
                    operator_value=item.operator_value,
                )

                if condition:
                    conditions.append(condition)
        else:
            return "", usage

        if not conditions:
            return "", usage
        elif len(conditions) == 1:
            return conditions[0], usage
        else:
            symbol = f' {metadata_info.logical_operator} '
            return f'({symbol.join(conditions)})', usage

    @classmethod
    def get_condition(
        cls, 
        metadata_name: str,
        metadata_type: str, 
        operator_name: str, 
        operator_value: str,
        table_name: str = 'knowledge_base_document_metadata'
    ) -> str:
        expression = ''
        if metadata_type == 'string':
            if operator_name == '=' or operator_name == 'is':
                expression = f"m.value = '{operator_value}'"
            elif operator_name == '!=':
                expression = f"m.value != '{operator_value}'"
            elif operator_name == 'contains':
                expression = f"m.value LIKE '%{operator_value}%'"
            elif operator_name == 'not contain':
                expression = f"m.value NOT LIKE '%{operator_value}%'"
            elif operator_name == 'starts with':
                expression = f"m.value LIKE '{operator_value}%'"
            elif operator_name == 'ends with':
                expression = f"m.value LIKE '%{operator_value}'"
            elif operator_name == 'is null':
                expression = "m.value IS NULL"
            elif operator_name == 'is not null':
                expression = "m.value IS NOT NULL"
            else:
                return ''
        elif metadata_type == 'number':
            if operator_name == '=':
                expression = f"m.value::float = {operator_value}"
            elif operator_name == '!=':
                expression = f"m.value::float != {operator_value}"
            elif operator_name == '>':
                expression = f"m.value::float > {operator_value}"
            elif operator_name == '<':
                expression = f"m.value::float < {operator_value}"
            elif operator_name == '>=':
                expression = f"m.value::float >= {operator_value}"
            elif operator_name == '<=':
                expression = f"m.value::float <= {operator_value}"
            elif operator_name == 'is null':
                expression = "m.value IS NULL"
            elif operator_name == 'is not null':
                expression = "m.value IS NOT NULL"
            else:
                return ''
        elif metadata_type == 'time':
            if operator_name == '=':
                expression = f"m.value::timestamp = TO_CHAR(TO_TIMESTAMP({int(operator_value)/1000.0}), 'YYYY-MM-DD HH24:MI')::timestamp"
            elif operator_name == '>':
                expression = f"m.value::timestamp > TO_CHAR(TO_TIMESTAMP({int(operator_value)/1000.0}), 'YYYY-MM-DD HH24:MI')::timestamp"
            elif operator_name == '<':
                expression = f"m.value::timestamp < TO_CHAR(TO_TIMESTAMP({int(operator_value)/1000.0}), 'YYYY-MM-DD HH24:MI')::timestamp"
            elif operator_name == 'is null':
                expression = "m.value IS NULL"
            elif operator_name == 'is not null':
                expression = "m.value IS NOT NULL"
            else:
                return ''
        else:
            return ''
        
        return (
            f"EXISTS (\n"
            f"SELECT 1\n"
            f"FROM {table_name} as m\n"
            f"WHERE m.document_id = d.id\n"
            f"  AND m.name = '{metadata_name}'\n"
            f"  AND {expression}\n"
            ")"
        )
from typing import Dict, Optional, List, Literal, Sequence, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage

from core.model.model_manager import ModelManager, ModelInstanceType
from core.model.model_instance import ModelInstanceRequestTimeout
from core.node.llm_invoker.base_utils import (
    render_messgaes, 
)
from core.agent.base_utils import (
    deserialize_history, 
    serialize_history, 
    truncate_history, 
)
from logger import get_logger

logger = get_logger('node')

async def allm_invoke(
    model_instance_provider: str,
    model_instance_config: Dict,
    model_parameters: Optional[Dict],
    stream: bool,
    prompt_messages: Sequence[Dict[str, str]],
    input_arguments: Optional[Dict],
    is_history: bool,
    chat_history: Optional[List],
    chat_history_depth: Optional[int],
    is_vision: bool,
    vision_images: Optional[List],
    vision_resolution: Optional[Literal['low', 'high']],
    is_structured_output: bool,
    structured_output_schema: Optional[Dict],
    **kwargs
) -> AsyncGenerator:
    # 获取LLM实例
    model_instance = ModelManager.get_model_instance(
        model_instance_provider,
        ModelInstanceType.LLM,
        **model_instance_config,
    )

    # 嵌入输入参数
    prompt_messages = render_messgaes(prompt_messages, input_arguments)

    # 转为List[BaseMessage]
    messages = deserialize_history(prompt_messages)

    # 拼接记忆
    history = []
    if is_history and chat_history:
        # 反序列化聊天历史
        history = deserialize_history(chat_history)
        # 聊天历史截断
        history = truncate_history(history, chat_history_depth)
        # 插入
        messages[1:1] = history

    # 根据视觉组装输入
    if is_vision and model_instance_config.get('is_support_vision', False) and vision_images:
        text = prompt_messages[-1].content
        images_content = [
            {
                "type": "image_url", 
                "image_url": image
            }
            for image in vision_images
        ]

        # 将视觉图片添加至提示词消息中
        if len(prompt_messages) >= 2:
            # 若原始消息超过2项，则添加至末尾
            messages[-1].content = [
                {
                    "type": "text", 
                    "text": text,
                }
            ]
            messages[-1].content.extend(images_content)
        else:
            # 若原始消息为1项，则添加至开头
            messages[0].content = [
                {
                    "type": "text", 
                    "text": text,
                }
            ]
            messages[0].content.extend(images_content)

    # 调用
    parameters = dict(model_parameters)
    if is_structured_output:
        parameters["format"] = structured_output_schema
    parameters["reasoning"] = False

    generator = model_instance.ainvoke_text_chat(
        prompt_messages=messages,
        model_parameters=parameters,
        stream=stream,
    )

    if stream:
        # 生成消息
        response_text = ''
        async for chunk in generator:
            response_text += chunk.assistant_message.content

            if chunk.finish_reason:
                # 生成完毕

                # 组装聊天历史
                if is_history:
                    if isinstance(messages[-1], HumanMessage):
                        history.append(
                            messages[-1]
                        )

                    history.append(
                        AIMessage(content=response_text)
                    )

                    # 聊天历史截断
                    history = truncate_history(history, chat_history_depth)

                logger.debug(f'LLM Invoke:\n{response_text}')

                output = {
                    'done': True,
                    'data': {
                        'assistant_message': chunk.assistant_message.content,
                        'chat_history': serialize_history(history) if is_history else []
                    },
                    'usage': chunk.usage.to_dict()
                }
            else:
                # 未生成完毕
                output = {
                    'done': False,
                    'data': {
                        'assistant_message': chunk.assistant_message.content,
                        'chat_history': None,
                    },
                    'usage': None,
                }

            yield output
    else:
        async for chunk in generator:
            # 组装聊天历史
            if is_history:
                if isinstance(messages[-1], HumanMessage):
                    history.append(
                        messages[-1]
                    )
            
                history.append(
                    AIMessage(content=chunk.assistant_message.content)
                )

                # 聊天历史截断
                history = truncate_history(history, chat_history_depth)

            logger.debug(f'LLM Invoke:\n{chunk.assistant_message.content}')

            output = {
                'done': True,
                'data': {
                    'assistant_message': chunk.assistant_message.content,
                    'chat_history': serialize_history(history) if is_history else []
                },
                'usage': chunk.usage.to_dict()
            }

            yield output
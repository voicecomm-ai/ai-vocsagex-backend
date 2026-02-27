from typing import Dict, Optional, Sequence

from jinja2 import Template, Undefined


def render_messgaes(
    messages: Sequence[Dict[str, str]],
    arguments: Optional[Dict],
):
    arguments = arguments or {}
    
    outputs = []
    for msg in messages:
        n_msg = msg.copy()
        if 'content' in msg:
            n_msg['content'] = Template(
                msg['content'], undefined=Undefined
            ).render(**arguments)
        outputs.append(n_msg)
    
    return outputs


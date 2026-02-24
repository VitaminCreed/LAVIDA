import random

USER_TEMPLATES_SHORT = [
"""Can you segment the anomalies in this {type}? Anomalies include {abnormal_example}. """,

"""Please segment the anomalies in this {type}. Abnormal cases contain {abnormal_example}. """,

"""Detect anomalies in this {type}. Target types: {abnormal_example}. """,

"""Monitor anomalies in this sceen. Anomalies include {abnormal_example}. """,

"""Segment abnormal events in this sceen. Abnormal events include {abnormal_example}. """
]

RESPONSE_TEMPLATES = [
    """It is {seg_placeholder}.""",
    """Sure, {seg_placeholder}.""",
    """Sure, it is {seg_placeholder}.""",
    """Sure, the segmentation result is {seg_placeholder}.""",
    """{seg_placeholder}.""",
]


def create_structured_template(
    path: str, 
    type: str,
    nframes: int = 1,
    anomaly_list: str = None,
):
    assert type in ['image', 'video']
    anomlies = ", ".join(anomaly_list)
    question = random.choice(USER_TEMPLATES_SHORT).format(
        type=type, abnormal_example=anomlies,
    )
    answer= random.choice(RESPONSE_TEMPLATES).format(
        seg_placeholder="<SEG>"  
    )
    
    if type == 'video':
        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': path, 'nframes': nframes},
                    {'type': 'text', 'text': question}
                ]
            },
            {
                'role': 'assistant',
                'content': [{
                    'type': 'text',
                    'text': answer,
                }]
            }
        ]
    else:
        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': path},
                    {'type': 'text', 'text': question}
                ]
            },
            {
                'role': 'assistant',
                'content': [{
                    'type': 'text',
                    'text': answer,
                }]
            }
        ]


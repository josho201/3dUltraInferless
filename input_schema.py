INPUT_SCHEMA = {
    "image_bytes": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://raw.githubusercontent.com/josho201/3dUltra/refs/heads/main/imgs/test2.png"]
    },
    'inference_steps': {
        'datatype': 'INT64',
        'required': True,
        'shape': [1],
        'example': [2]
    },
    'controlet': {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [0.8]
    },
    'guidance': {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [1.25]
    },
    'prompt': {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Photorealistic portrait of a cute asleep newborn baby, closed eyes, soft light, DSLR, 85mm lens"]
    }
}

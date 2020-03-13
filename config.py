config_craft = {
    'trained_model': './CLOVA_CRAFT/weights/craft_mlt_25k.pth',
    'text_threshold': 0.7,
    'low_text': 0.4,
    'link_threshold': 0.4,
    'canvas_size': 1280,
    'mag_ratio': 1.5,
    'poly': False,
    'show_time': False,
    'test_folder': './data/test_folder/',
    'refine': False,
    'refiner_model': './CLOVA_CRAFT/weights/craft_refiner_CTW1500.pth'
}

config_ocr = {
    'image_folder': './data/test_folder/',
    'workers': 4,
    'batch_size': 1,
    'saved_model': './CLOVA_OCR/weights/TPS-ResNet-BiLSTM-Attn-Seed190702001/best_accuracy.pth',

    'batch_max_length': 25,
    'imgH': 32,
    'imgW': 100,
    'rgb': False,
    'character': '0123456789abcdefghijklmnopqrstuvwxyz',
    'sensitive': True,
    'PAD': False,

    'Transformation': 'TPS',
    'FeatureExtraction': 'ResNet',
    'SequenceModeling': 'BiLSTM',
    'Prediction': 'Attn',
    'num_fiducial': 20,
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256
}
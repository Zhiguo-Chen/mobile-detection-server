from src.datasets.object_detection_dataset import ObjectDetectionDataset
DATASETS = {
    'object_detection': ObjectDetectionDataset
}


def get_dataset(dataset_type):
    if dataset_type not in DATASETS:
        raise ValueError('"{}" is not a valid data type'.format(dataset_type))
    return DATASETS[dataset_type]

import torch
from workflow.torch import Dataset, Datastream


image_dataframe = pd.DataFrame([
    dict(image_id='1', label=1, path='image1.png'),
    dict(image_id='2', label=0, path='image2.png'),
    dict(image_id='3', label=1, path='image3.png'),
])

dataset = (
    Dataset.from_dataframe(image_dataframe)
    .map(lambda row: (
        row['image_id'],
        read_image(row['path']),
        row['label'],
    ))
    .map(lambda image_id, image, class: dict(
        image_id=image_id,
        image=preprocess_image(image),
        label=torch.as_tensor(label),
    ))
)

datastream = Datastream.interleave([
    Datastream(dataset),
    some_other_datastream,
    yet_another_datastream,
])

data_loader = datastream.data_loader(batch_size=8, num_workers=4)

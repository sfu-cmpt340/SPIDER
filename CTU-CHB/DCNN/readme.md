## How to Use:
#### Run the following in order:

- Install requirements: `pip install -r requirements.txt`
- Process, load and label: `python process_load_label.py`
- Generate test/train split (takes a min or more): `python generate.py <imagetype>` (where imagetype is either `spectrogram`, `cwt`, or `recurrence_plots`)
- Train Model: `python train.py <num of epochs>`, 10-50 epochs recommended

Sample trained models are in the folder 'Pre-trained Samples'. Run `python test.py` to test these models.

To load and use: 

```
with open('model_architecture.json', 'r') as json_file:
    architecture = json.load(json_file)
model = tf.keras.models.model_from_json(architecture)
model.load_weights('model.weights.h5')
```
## How to Use:
- Install requirements: `pip install -r requirements.txt`
- Process, load and label: `python process_load_label.py`
- Generate test/train split (takes a min or more): `python generate.py`
- Train Model: `python train.py <num of epochs>`, GPU and 10-50 epochs recommended

Model is already trained and saved seperately as model.weights.h5 and model.architecture.json. 

To load and use: 

```
with open('model_architecture.json', 'r') as json_file:
    architecture = json.load(json_file)
model = tf.keras.models.model_from_json(architecture)
model.load_weights('model_weights.h5')
```
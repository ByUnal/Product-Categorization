![Wildfire-cover](https://images.ctfassets.net/3viuren4us1n/4jw95A7JsM8bx6jPGCBSka/9c16251949c69108d0702d7fc0ce8042/product_categorization.jpg)
<h2 align="center">Product Categorization with Neural Networks</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

## Overview

The API provides multi-label classification model for product categorization.

## Data Installation and Preparation
Firstly, create ```data``` folder in the directory. Then, you need to download the 
[dataset](https://drive.google.com/file/d/1jRPAJuJqQmaZZUiciDdmP0FZ3VGhQwWc/view?usp=sharing). Next, put it under the "data" folder.
You can see the steps I followed while preparing the data for the training below. Open the terminal in the project's directory first.
Then go inside "src" folder. Run the code below, first.
```
python data_processing.py
```
It will save the extract new CSV file which is prepared for the training.
Dataset is ready for the training and it will be extracted to "categories.csv".

Before training the model, you can examine the data in detail thanks to notebook I shared.

## Running the API

### via Docker
Build the image inside the Dockerfile's directory
```commandline
docker build -t prod-cat .
```
Then for running the image in local network
```commandline
docker run --network host --name product_categorization prod-cat
```
Finally, you can use the API by sending request in JSON format:
```bash
http://localhost:5000/prediction
```

### via Python in Terminal

Open the terminal in the project's directory.
Install the requirements first.
```commandline
pip install -r requirements.txt
```
Then, run the main.py file
```commandline
python app.py
```

Finally, you can use the API by sending request to

```bash
http://localhost:5000/prediction
```

### Example Usage
#### Product Category Prediction

Send the JSON query by using cURL, Postman or any other tools/commands.
```
curl --location --request POST 'http://localhost:5000/prediction' 
--data-raw '{ "product_name": "Stackable Water Bottle Storage Rack Best Water Jugs 5 Gallon Organizer. Jug Holder for Kitchen, Cabinet and Office Organizing. Reinforced Polypropylene. 3 Plus Shelf, Silver" }'
```

Then result would be something like this
```
{
    "categories": "shoes & accessories:men's clothing:shirts:t-shirts > 2 piece set > 3 piece set"
}
```

## Train Model
Training can be done by using different parameters by using environment variable.
```commandline
python train.py --learnin_rate 0.3 --train_size 0.7 --batch_size 128
```

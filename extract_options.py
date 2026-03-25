import csv
import json
import os

filepath = r'd:\bad thingd\1\StockAgent\Remido\DoshaPredictor\dataset.csv'
options = {}

with open(filepath, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    features = headers[:-1]
    
    # Initialize options dict
    for col in features:
        options[col] = set()
        
    for row in reader:
        for i, val in enumerate(row[:-1]):
            options[features[i]].add(val)

# Convert sets to sorted lists
for col in options:
    options[col] = sorted(list(options[col]))

output_path = r'd:\bad thingd\1\StockAgent\Remido\DoshaPredictor\feature_options.json'
with open(output_path, 'w') as f:
    json.dump(options, f, indent=4)

print(f"Feature options extracted to {output_path}")

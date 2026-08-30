# TrafficCLIP Dataset

This folder contains the dataset used by TrafficCLIP.

## Folder Structure

```text
data/
├── raw/
├── processed/
└── splits/
```

### raw/

Store the original PCAP files here.

Example:

```text
data/raw/
├── example1.pcap
├── example2.pcap
└── ...
```

The original dataset files should not be modified.

### processed/

Store the traffic images generated from the PCAP files here.

Example:

```text
data/processed/
├── flow_000001.png
├── flow_000002.png
└── ...
```

### splits/

Contains the dataset split CSV files:

```text
train.csv
val.csv
test.csv
```

Each CSV contains:

```text
image_path,label
```

Example:

```csv
image_path,label
data/processed/flow_000001.png,Class1
data/processed/flow_000002.png,Class2
```

The actual class names and image paths will be generated after preprocessing the dataset.


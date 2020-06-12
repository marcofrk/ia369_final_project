class ObjectsDetectionTFLiteSSD:
    def __init__(self):
        self.model = None
        self.label = None

    def gather_data(self):
        self.model = "../data/model/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
        self.label = "../data/model/coco_labels.txt"

    def load_labels(self, label_path):
        with open(label_path) as f:
            labels = {}
            for line in f.readlines():
                m = re.match(r"(\d+)\s+(\w+)", line.strip())
                labels[int(m.group(1))] = m.group(2)
            return labels

    def start(self):
        self.gather_data()
        self.label = self.load_labels(self.label)

        if self.model is None or self.label is None:
            print("Your model/label parameters is invalid.")

    def run(self):
        self.start()

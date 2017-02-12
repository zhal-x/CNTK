import cntk as C
import numpy as np
from cntk.io import MinibatchSource, HTKDeserializers, StreamDef, StreamDefs
from cntk.blocks import LSTM, Placeholder, Input
from cntk.layers import Recurrence, Dense, BatchNormalization
from cntk.models import Sequential, For

import os, sys
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Speech", "AN4", "Data")

def test_htk_deserializers():
    mbsize = 640
    epoch_size = 1000 * mbsize
    lr = [0.001]

    feature_dim = 33
    num_classes = 132
    context = 2

    os.chdir(data_path)

    features_file = "glob_0000.scp"
    labels_file = "glob_0000.mlf"
    label_mapping_file = "state.list"

    reader = MinibatchSource(HTKDeserializers(label_mapping_file, StreamDefs(
        features = StreamDef(features_file, shape=feature_dim, context=(context,context)),
        labels   = StreamDef(labels_file,   shape=num_classes))))

    features = C.input_variable(((2*context+1)*feature_dim))
    labels = C.input_variable((num_classes))

    model = Sequential([BatchNormalization(),
                        For(range(3), lambda : Recurrence(LSTM(256))),
                        Dense(num_classes)])
    z = model(features)
    ce = C.cross_entropy_with_softmax(z, labels)
    errs = C.classification_error    (z, labels)

    learner = C.adam_sgd(z.parameters,
                    lr=C.learning_rate_schedule(lr, C.UnitType.sample, epoch_size),
                    momentum=C.momentum_as_time_constant_schedule(1000),
                    low_memory=True,
                    gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)
    trainer = C.Trainer(z, (ce, errs), learner)

    input_map={ features: reader.streams.features, labels: reader.streams.labels }

    pp = C.ProgressPrinter(freq=0)
    # just run and verify it doesn't crash
    for i in range(3):
        mb_data = reader.next_minibatch(mbsize, input_map=input_map)
        trainer.train_minibatch(mb_data)
        pp.update_with_trainer(trainer, with_metric=True)
    os.chdir(abs_path)

test_htk_deserializers()

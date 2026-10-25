#!/usr/bin/env python3
"""Train the five gesture classes shown in the supplied figure.

Install (Python 3.10-3.12 suggested):
    python -m pip install tensorflow numpy
Train:
    python train_mpu6050_gestures.py --csv MPU6050_Gesture_Features_1000_Synthetic.csv --out gesture_model
Check data/splits without TensorFlow:
    python train_mpu6050_gestures.py --csv MPU6050_Gesture_Features_1000_Synthetic.csv --check-data

Input CSV: one row per 50-reading window, with 24 statistical features.
Input order: mean, population std (ddof=0), min, max for ax, ay, az, gx, gy, gz.
Acceleration in g, gyro in degrees/second; gravity retained. Nominal 50Hz.
IDs, timestamps, data_source and labels are excluded from model inputs.

Classes/actions:
  0 tilt_left  -> OFF
  1 tilt_right -> ON
  2 up         -> increase brightness
  3 down       -> decrease brightness
  4 idle       -> preserve current state

Outputs:
  model.h: full INT8 byte array plus 24-feature normalization/quantization helpers
  model_int8.tflite, model.keras, training_report.json,
  test_predictions.csv, confusion_matrix.csv, split_assignments.csv.

ESP32 / EloquentTinyML 3.x integration:
  #include "model.h"          // before EloquentTinyML
  #include <tflm_esp32.h>
  #include <eloquent_tinyml.h>
  Eloquent::TF::Sequential<2, 24 * 1024> tf;
  // In setup, setNumInputs(24), setNumOutputs(5), add FullyConnected/Softmax,
  // then check tf.begin(model_data).isOk(). Arena size needs device validation.
  // In loop AFTER extracting 24 window features:
  int8_t q[24];
  gesture_prepare_input(features, q);
  if (tf.predict(q).isOk()) { int label = tf.classification; }
Do not feed six raw values or normalize twice. The earlier six-input/three-LED
sketch requires changes for 24 inputs and five gesture commands.

Synthetic data demonstrate the pipeline, not real gesture recognition.
Up/down may overlap because these features discard temporal order. Do not
increase reported accuracy by changing the held-out test set or using IDs.
For deployment, collect real labeled recordings and validate unwanted commands.
Use idle re-arming/cooldown in the lighting controller to avoid repeated actions.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
STATS = ['mean', 'std', 'min', 'max']
FEATURES = [f'{axis}_{stat}' for axis in AXES for stat in STATS]
LABELS = ['tilt_left', 'tilt_right', 'up', 'down', 'idle']
ACTIONS = ['OFF', 'ON', 'BRIGHTNESS_UP', 'BRIGHTNESS_DOWN', 'KEEP_STATE']


def load_data(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        # Accept Arduino comment lines; still reject repeated CSV headers.
        reader = csv.DictReader(line for line in file if not line.lstrip().startswith('#'))
        missing = set(FEATURES + ['recording_id', 'label_id', 'label']) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing CSV columns: {sorted(missing)}')
        records = list(reader)
    if not records:
        raise ValueError('CSV contains no data rows')
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in records], dtype=np.float32)
    y = np.asarray([int(row['label_id']) for row in records], dtype=np.int32)
    if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= 5):
        raise ValueError('Nonfinite features or label_id outside 0..4')
    groups = []
    for i, row in enumerate(records):
        if row['label'] != LABELS[y[i]]:
            raise ValueError(f'Row {i+2}: label_id does not match label text')
        if not row['recording_id'].strip():
            raise ValueError(f'Row {i+2}: empty recording_id')
        # JSON tuple encoding prevents ambiguous combined identifiers.
        groups.append(json.dumps([row.get('session_id', ''), row['recording_id']]))
    for axis in range(6):
        mean, std, low, high = x[:, axis*4:axis*4+4].T
        if np.any(std < 0) or np.any(low > mean + 1e-4) or np.any(mean > high + 1e-4):
            raise ValueError(f'Invalid mean/std/min/max for {AXES[axis]}')
    return x, y, np.asarray(groups), records


def split_by_recording(y, groups, seed):
    rng = np.random.default_rng(seed)
    assignments = {}
    for group in np.unique(groups):
        if len(np.unique(y[groups == group])) != 1:
            raise ValueError(f'Recording has mixed labels: {group}. Add session_id if IDs repeat across sessions.')
    for label_id, label in enumerate(LABELS):
        ids = np.unique(groups[y == label_id])
        if len(ids) < 5:
            raise ValueError(f'{label}: at least five independent recordings are required')
        rng.shuffle(ids)
        n_test = max(1, int(round(.2 * len(ids))))
        n_val = max(1, int(round(.2 * len(ids))))
        for i, group in enumerate(ids):
            assignments[str(group)] = ('test' if i < n_test else
                                       'validation' if i < n_test+n_val else 'train')
    splits = {name: np.asarray([i for i, group in enumerate(groups) if assignments[str(group)] == name])
              for name in ['train', 'validation', 'test']}
    for name, indices in splits.items():
        if set(y[indices].tolist()) != set(range(5)):
            raise ValueError(f'{name} does not contain all five classes')
    return splits, assignments


def quantize(values, scale, zero):
    # Round half away from zero, matching generated C++ roundf().
    scaled = values / scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + .5)
    return np.clip(rounded + zero, -128, 127).astype(np.int8)


def write_header(path, content, mean, std, input_quant, output_quant):
    def cfloat(value):
        return f'{float(value):.9e}f'
    lines = [
        '// Generated by train_mpu6050_gestures.py. Requires hardware validation.',
        '// Feature order: ' + ','.join(FEATURES),
        '#pragma once', '#include <stdint.h>', '#include <math.h>', '',
        'constexpr int GESTURE_NUM_INPUTS = 24;',
        'constexpr int GESTURE_NUM_CLASSES = 5;',
        'alignas(16) const unsigned char model_data[] = {',
    ]
    for offset in range(0, len(content), 12):
        lines.append('    ' + ', '.join(f'0x{value:02X}' for value in content[offset:offset+12]) + ',')
    lines += [
        '};', 'const unsigned int model_data_len = sizeof(model_data);',
        'const float gesture_mean[24] = {' + ', '.join(map(cfloat, mean)) + '};',
        'const float gesture_std[24] = {' + ', '.join(map(cfloat, std)) + '};',
        f'const float gesture_input_scale = {cfloat(input_quant[0])};',
        f'const int gesture_input_zero_point = {int(input_quant[1])};',
        f'const float gesture_output_scale = {cfloat(output_quant[0])};',
        f'const int gesture_output_zero_point = {int(output_quant[1])};',
        'const char* const gesture_labels[5] = {' + ', '.join(json.dumps(s) for s in LABELS) + '};',
        'const char* const gesture_actions[5] = {' + ', '.join(json.dumps(s) for s in ACTIONS) + '};', '',
        '// Input is 24 unnormalized WINDOW FEATURES, not six raw readings.',
        'inline void gesture_prepare_input(const float features[24], int8_t result[24]) {',
        '    for (int i = 0; i < 24; ++i) {',
        '        float z = (features[i] - gesture_mean[i]) / gesture_std[i];',
        '        float q = roundf(z / gesture_input_scale) + gesture_input_zero_point;',
        '        q = fmaxf(-128.0f, fminf(127.0f, q));',
        '        result[i] = static_cast<int8_t>(q);', '    }', '}', '',
        'inline float gesture_output_value(int8_t value) {',
        '    return (static_cast<int>(value) - gesture_output_zero_point) * gesture_output_scale;',
        '}', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=Path('MPU6050_Gesture_Features_1000_Synthetic.csv'))
    parser.add_argument('--out', type=Path, default=Path('gesture_model'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    x, y, groups, records = load_data(args.csv)
    splits, assignment = split_by_recording(y, groups, args.seed)
    print('Features:', len(FEATURES), 'Rows:', len(y))
    print('Class counts:', dict(zip(LABELS, np.bincount(y, minlength=5).tolist())))
    print('Sequence-disjoint split:', {name: len(indices) for name, indices in splits.items()})
    synthetic = any('synthetic' in row.get('data_source', '').lower() for row in records)
    if synthetic:
        print('NOTICE: Synthetic data. Scores will not measure real gesture accuracy.')
    if args.check_data:
        print('Input/schema/group checks passed. TensorFlow was not imported.')
        return
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit('Install dependencies: python -m pip install tensorflow numpy') from exc
    tf.keras.utils.set_random_seed(args.seed)
    train, val, test = (splits[name] for name in ['train', 'validation', 'test'])
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0, ddof=0), 1e-6)
    normalized = ((x - mean) / std).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(24,), name='gesture_features'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax', name='gesture_scores'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(normalized[train], y[train],
                        validation_data=(normalized[val], y[val]),
                        epochs=args.epochs, batch_size=32, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=25, restore_best_weights=True)])
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / 'model.keras')
    float_predictions = np.argmax(model.predict(normalized[test], verbose=0), axis=1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for row in normalized[train]:
            yield [row.reshape(1, 24)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    (args.out / 'model_int8.tflite').write_bytes(content)
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if (inp['dtype'] != np.int8 or out['dtype'] != np.int8 or
            inp['shape'].tolist() != [1, 24] or out['shape'].tolist() != [1, 5]):
        raise RuntimeError('Expected INT8 tensors [1,24] and [1,5]')
    iq, oq = inp['quantization'], out['quantization']
    if iq[0] <= 0 or oq[0] <= 0:
        raise RuntimeError('Invalid quantization parameters')
    predictions = []
    quantized_inputs = quantize(normalized[test], *iq)
    for row in quantized_inputs:
        interpreter.set_tensor(inp['index'], row.reshape(1, 24))
        interpreter.invoke()
        predictions.append(int(np.argmax(interpreter.get_tensor(out['index'])[0])))
    predictions = np.asarray(predictions, dtype=np.int32)
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (y[test], predictions), 1)
    recall = np.diag(confusion) / confusion.sum(axis=1)
    write_header(args.out / 'model.h', content, mean, std, iq, oq)
    report = {
        'tensorflow_version': tf.__version__, 'seed': args.seed,
        'contains_synthetic_data': synthetic,
        'input_csv': str(args.csv), 'features': FEATURES,
        'class_mapping': [{'id': i, 'label': label, 'action': ACTIONS[i]} for i, label in enumerate(LABELS)],
        'architecture': '24 -> Dense32/ReLU -> Dense16/ReLU -> Dense5/Softmax',
        'split_method': 'Per-class shuffled recordings, approximately 60/20/20',
        'split_rows': {name: len(indices) for name, indices in splits.items()},
        'normalization_mean': mean.tolist(), 'normalization_std': std.tolist(),
        'epochs_run': len(history.history['loss']),
        'keras_test_accuracy': float(np.mean(float_predictions == y[test])),
        'int8_test_accuracy': float(np.mean(predictions == y[test])),
        'int8_macro_recall': float(recall.mean()),
        'per_class_recall': dict(zip(LABELS, recall.tolist())),
        'confusion_matrix_true_rows_predicted_columns': confusion.tolist(),
        'quantization': {'input': [float(iq[0]), int(iq[1])], 'output': [float(oq[0]), int(oq[1])]},
        'model_bytes': len(content),
        'history': {key: [float(value) for value in values] for key, values in history.history.items()},
        'limitations': ['Synthetic performance is not real-sensor validation.',
                       'Summary statistics discard temporal order, potentially confusing up/down.',
                       'Desktop TFLite evaluation does not verify ESP32/TFLM kernel compatibility.',
                       'Lighting commands need idle re-arming and validation of false activations.'],
    }
    (args.out / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.out / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label/predicted_label'] + LABELS)
        writer.writerows([[LABELS[i]] + row.tolist() for i, row in enumerate(confusion)])
    with (args.out / 'split_assignments.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'recording_id', 'split'])
        for group, split in sorted(assignment.items()):
            writer.writerow(json.loads(group) + [split])
    with (args.out / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['csv_data_row', 'recording_id', 'true_label', 'predicted_label', 'correct'])
        for index, predicted in zip(test, predictions):
            writer.writerow([int(index)+1, records[index]['recording_id'],
                             LABELS[y[index]], LABELS[predicted], int(y[index] == predicted)])
    print('Float test accuracy:', report['keras_test_accuracy'])
    print('INT8 test accuracy:', report['int8_test_accuracy'])
    print('Per-class recall:', report['per_class_recall'])
    print('Confusion matrix (true rows, predicted columns):\n', confusion)
    print('Exported model.h:', (args.out / 'model.h').resolve())


if __name__ == '__main__':
    main()
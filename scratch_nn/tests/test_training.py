"""End-to-end tests: optimizers, the training loop, data handling, metrics,
serialization, and XOR convergence."""

import json
import math
import os
import pickle
import tempfile
import unittest
import xml.dom.minidom

from scratch_nn.data import (LabelEncoder, MinMaxScaler, StandardScaler,
                             batch_iterator, from_one_hot, load_csv, one_hot,
                             save_csv, train_test_split, train_val_test_split,
                             xor_dataset)
from scratch_nn.layers import Dense, Dropout
from scratch_nn.losses import MeanSquaredError
from scratch_nn.metrics import (accuracy, classification_report,
                                confusion_matrix, confusion_matrix_svg,
                                f1_score, mean_absolute_error, precision,
                                r_squared, recall, root_mean_squared_error,
                                save_confusion_matrix_svg)
from scratch_nn.model import Sequential
from scratch_nn.optimizers import (SGD, AdaGrad, Adam, Momentum, RMSProp,
                                   get_optimizer)
from scratch_nn.serialization import (load_model, load_pickle, save_model,
                                      save_pickle)
from scratch_nn.tensor import Tensor
from scratch_nn.training import (EarlyStopping, LearningRateScheduler,
                                 ModelCheckpoint, cosine_decay, step_decay)
from scratch_nn.utils import clip_gradient, gradient_norm, set_seed


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------

class TestOptimizers(unittest.TestCase):

    def test_sgd_applies_the_update_rule(self):
        param = Tensor([1.0, 2.0])
        grad = Tensor([0.5, 1.0])
        opt = SGD(learning_rate=0.1)
        opt.build([param])
        opt.step([param], [grad])
        # W <- W - lr*g
        self.assertAlmostEqual(param.data[0], 1.0 - 0.1 * 0.5)
        self.assertAlmostEqual(param.data[1], 2.0 - 0.1 * 1.0)

    def test_sgd_moves_parameters_downhill(self):
        param = Tensor([5.0])
        opt = SGD(0.1)
        opt.build([param])
        for _ in range(10):
            # Gradient of f(w) = w^2 is 2w.
            opt.step([param], [Tensor([2.0 * param.data[0]])])
        self.assertLess(abs(param.data[0]), 5.0)

    def test_momentum_accumulates_velocity(self):
        param = Tensor([0.0])
        opt = Momentum(learning_rate=0.1, momentum=0.9)
        opt.build([param])
        grad = Tensor([1.0])
        opt.step([param], [grad])
        after_one = param.data[0]
        opt.step([param], [Tensor([1.0])])
        after_two = param.data[0]
        # v = 1 then v = 1.9, so the second step is larger than the first.
        self.assertAlmostEqual(after_one, -0.1)
        self.assertLess(after_two - after_one, after_one)

    def test_nesterov_variant_runs(self):
        param = Tensor([1.0])
        opt = Momentum(0.1, 0.9, nesterov=True)
        opt.build([param])
        opt.step([param], [Tensor([1.0])])
        self.assertLess(param.data[0], 1.0)

    def test_adam_bias_correction_gives_a_full_first_step(self):
        param = Tensor([0.0])
        opt = Adam(learning_rate=0.1)
        opt.build([param])
        opt.step([param], [Tensor([1.0])])
        # With correction the first step is ~lr; without it, ~lr/10.
        self.assertAlmostEqual(abs(param.data[0]), 0.1, places=4)

    def test_adam_normalizes_by_gradient_scale(self):
        # Two parameters with very different gradient magnitudes should take
        # steps of comparable size.
        small, large = Tensor([0.0]), Tensor([0.0])
        opt = Adam(learning_rate=0.1)
        opt.build([small, large])
        for _ in range(5):
            opt.step([small, large], [Tensor([0.001]), Tensor([1000.0])])
        self.assertAlmostEqual(abs(small.data[0]), abs(large.data[0]), places=3)

    def test_adam_state_grows_with_steps(self):
        param = Tensor([1.0])
        opt = Adam(0.01)
        opt.build([param])
        for _ in range(5):
            opt.step([param], [Tensor([0.5])])
        self.assertEqual(opt.iterations, 5)

    def test_amsgrad_runs(self):
        param = Tensor([1.0])
        opt = Adam(0.01, amsgrad=True)
        opt.build([param])
        opt.step([param], [Tensor([0.5])])
        self.assertLess(param.data[0], 1.0)

    def test_rmsprop_and_adagrad_update(self):
        for opt_cls in (RMSProp, AdaGrad):
            param = Tensor([1.0])
            opt = opt_cls(learning_rate=0.1)
            opt.build([param])
            opt.step([param], [Tensor([1.0])])
            self.assertLess(param.data[0], 1.0, f"{opt_cls.__name__} did not update")

    def test_zero_gradient_leaves_parameters_alone(self):
        param = Tensor([1.0, 2.0])
        opt = SGD(0.1)
        opt.build([param])
        opt.step([param], [Tensor([0.0, 0.0])])
        self.assertEqual(param.tolist(), [1.0, 2.0])

    def test_lr_scale_is_applied(self):
        param = Tensor([1.0])
        opt = SGD(0.1)
        opt.build([param])
        opt.lr_scale = 0.5
        self.assertAlmostEqual(opt.current_lr, 0.05)
        opt.step([param], [Tensor([1.0])])
        self.assertAlmostEqual(param.data[0], 1.0 - 0.05)

    def test_mismatched_lengths_rejected(self):
        opt = SGD(0.1)
        with self.assertRaises(ValueError):
            opt.step([Tensor([1.0])], [])

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            SGD(learning_rate=-0.1)
        with self.assertRaises(ValueError):
            Momentum(0.1, momentum=1.5)
        with self.assertRaises(ValueError):
            Adam(0.1, beta1=1.0)

    def test_registry(self):
        self.assertIsInstance(get_optimizer("adam"), Adam)
        self.assertIsInstance(get_optimizer("sgd"), SGD)
        self.assertTrue(get_optimizer("nesterov").nesterov)
        with self.assertRaises(ValueError):
            get_optimizer("not_an_optimizer")


# ---------------------------------------------------------------------------
# XOR — the acceptance criterion
# ---------------------------------------------------------------------------

class TestXORConvergence(unittest.TestCase):

    def _train(self, optimizer, epochs, activation="tanh"):
        set_seed(42)
        X, y = xor_dataset()
        model = Sequential([
            Dense(2, 8, activation=activation, initialization="xavier"),
            Dense(8, 8, activation=activation, initialization="xavier"),
            Dense(8, 1, activation="sigmoid", initialization="xavier"),
        ])
        model.compile(loss="binary_cross_entropy", optimizer=optimizer,
                      metrics=["accuracy"])
        history = model.fit(X, y, epochs=epochs, batch_size=4, verbose=0)
        return model, history, X, y

    def test_xor_is_learned_with_adam(self):
        model, history, X, y = self._train(Adam(0.05), 400)
        self.assertEqual(history.last("accuracy"), 1.0)
        self.assertLess(history.last("loss"), 0.05)

    def test_xor_predictions_are_correct(self):
        model, _, X, y = self._train(Adam(0.05), 400)
        predictions = model.predict(X)
        for i, target in enumerate(y):
            label = 1 if predictions.data[i] >= 0.5 else 0
            self.assertEqual(label, int(target[0]),
                             f"input {X[i]} predicted {label}, expected {target[0]}")

    def test_xor_is_learned_with_plain_sgd(self):
        _, history, _, _ = self._train(SGD(0.5), 3000)
        self.assertEqual(history.last("accuracy"), 1.0)

    def test_xor_is_learned_with_relu(self):
        _, history, _, _ = self._train(Adam(0.05), 600, activation="relu")
        self.assertEqual(history.last("accuracy"), 1.0)

    def test_single_layer_cannot_learn_xor(self):
        # The historically important negative result: XOR is not linearly
        # separable, so a network with no hidden layer must fail.
        set_seed(42)
        X, y = xor_dataset()
        model = Sequential([Dense(2, 1, activation="sigmoid")])
        model.compile(loss="binary_cross_entropy", optimizer=Adam(0.1),
                      metrics=["accuracy"])
        history = model.fit(X, y, epochs=500, batch_size=4, verbose=0)
        self.assertLessEqual(history.last("accuracy"), 0.75)

    def test_loss_decreases_monotonically_overall(self):
        _, history, _, _ = self._train(Adam(0.05), 200)
        losses = history["loss"]
        self.assertLess(losses[-1], losses[0] * 0.1)


# ---------------------------------------------------------------------------
# training loop mechanics
# ---------------------------------------------------------------------------

class TestTrainingLoop(unittest.TestCase):

    def _simple_problem(self):
        set_seed(7)
        X = [[float(i), float(i) * 2] for i in range(40)]
        y = [[1.0] if i > 20 else [0.0] for i in range(40)]
        return X, y

    def test_history_records_every_epoch(self):
        X, y = self._simple_problem()
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.01), metrics=["accuracy"])
        history = model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        self.assertEqual(len(history["loss"]), 10)
        self.assertEqual(len(history["accuracy"]), 10)
        self.assertEqual(history.epochs, 10)

    def test_validation_metrics_are_recorded(self):
        X, y = self._simple_problem()
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.01), metrics=["accuracy"])
        history = model.fit(X, y, epochs=5, batch_size=8,
                            validation_data=(X[:10], y[:10]), verbose=0)
        self.assertIn("val_loss", history)
        self.assertIn("val_accuracy", history)

    def test_validation_split(self):
        X, y = self._simple_problem()
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.01))
        history = model.fit(X, y, epochs=3, validation_split=0.25, verbose=0)
        self.assertIn("val_loss", history)

    def test_batch_sizes_all_work(self):
        X, y = self._simple_problem()
        for batch_size in (1, 8, 40, None):
            set_seed(7)
            model = Sequential([Dense(2, 4, activation="tanh"),
                                Dense(4, 1, activation="sigmoid")])
            model.compile(loss="mse", optimizer=SGD(0.01))
            history = model.fit(X, y, epochs=3, batch_size=batch_size, verbose=0)
            self.assertEqual(len(history["loss"]), 3)

    def test_training_is_reproducible_with_a_seed(self):
        X, y = self._simple_problem()
        results = []
        for _ in range(2):
            set_seed(123)
            model = Sequential([Dense(2, 5, activation="tanh"),
                                Dense(5, 1, activation="sigmoid")])
            model.compile(loss="mse", optimizer=Adam(0.01))
            history = model.fit(X, y, epochs=5, batch_size=8, verbose=0)
            results.append(history["loss"])
        self.assertEqual(results[0], results[1])

    def test_weights_actually_change(self):
        X, y = self._simple_problem()
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.01))
        before = [list(p.data) for p in model.parameters()]
        model.fit(X, y, epochs=5, batch_size=8, verbose=0)
        after = [list(p.data) for p in model.parameters()]
        self.assertNotEqual(before, after)

    def test_gradient_clipping_caps_the_norm(self):
        grads = [Tensor([3.0, 4.0])]      # norm 5
        before = clip_gradient(grads, max_norm=1.0)
        self.assertAlmostEqual(before, 5.0)
        self.assertAlmostEqual(gradient_norm(grads), 1.0, places=10)
        # Direction is preserved: the ratio between components is unchanged.
        self.assertAlmostEqual(grads[0].data[1] / grads[0].data[0], 4.0 / 3.0)

    def test_clipping_leaves_small_gradients_alone(self):
        grads = [Tensor([0.3, 0.4])]      # norm 0.5
        clip_gradient(grads, max_norm=1.0)
        self.assertAlmostEqual(grads[0].data[0], 0.3)

    def test_empty_dataset_rejected(self):
        model = Sequential([Dense(2, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        with self.assertRaises(ValueError):
            model.fit([], [], epochs=1, verbose=0)

    def test_uncompiled_model_rejected(self):
        model = Sequential([Dense(2, 1)])
        with self.assertRaises(RuntimeError):
            model.fit([[0.0, 0.0]], [[0.0]], epochs=1, verbose=0)


class TestCallbacks(unittest.TestCase):

    def _overfittable(self):
        set_seed(11)
        X = [[float(i) / 20.0] for i in range(20)]
        y = [[float(i % 2)] for i in range(20)]     # unlearnable noise
        return X, y

    def test_early_stopping_halts_training(self):
        X, y = self._overfittable()
        model = Sequential([Dense(1, 30, activation="relu"),
                            Dense(30, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.05))
        stopper = EarlyStopping(monitor="val_loss", patience=3, verbose=False)
        history = model.fit(X, y, epochs=200, batch_size=4,
                            validation_data=(X, y), early_stopping=stopper,
                            verbose=0)
        self.assertLess(history.epochs, 200)
        self.assertTrue(history.stopped_early)

    def test_early_stopping_restores_the_best_weights(self):
        X, y = self._overfittable()
        model = Sequential([Dense(1, 20, activation="relu"),
                            Dense(20, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=Adam(0.05))
        stopper = EarlyStopping(monitor="val_loss", patience=3,
                                restore_best_weights=True, verbose=False)
        model.fit(X, y, epochs=100, batch_size=4, validation_data=(X, y),
                  early_stopping=stopper, verbose=0)
        final = model.evaluate(X, y)["loss"]
        # The restored weights must be at least as good as the best seen.
        self.assertLessEqual(final, stopper.best + 1e-6)

    def test_min_delta_ignores_trivial_improvements(self):
        stopper = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.5,
                                verbose=False)
        model = Sequential([Dense(1, 2, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        history_stub = type("H", (), {"stopped_early": False, "best_epoch": None})()
        stopper.on_train_begin(model, history_stub)
        # Improvements smaller than min_delta must not reset the counter.
        stopper.on_epoch_end(1, {"val_loss": 1.0}, model, history_stub)
        stopper.on_epoch_end(2, {"val_loss": 0.99}, model, history_stub)
        stop = stopper.on_epoch_end(3, {"val_loss": 0.98}, model, history_stub)
        self.assertTrue(stop)

    def test_checkpoint_writes_a_file(self):
        X, y = self._overfittable()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.json")
            model = Sequential([Dense(1, 5, activation="tanh"),
                                Dense(5, 1, activation="sigmoid")])
            model.compile(loss="mse", optimizer=Adam(0.01))
            model.fit(X, y, epochs=5, batch_size=4, validation_data=(X, y),
                      callbacks=[ModelCheckpoint(path, verbose=False)], verbose=0)
            self.assertTrue(os.path.exists(path))
            self.assertIsNotNone(load_model(path))

    def test_learning_rate_scheduler_changes_the_rate(self):
        X, y = self._overfittable()
        model = Sequential([Dense(1, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        scheduler = LearningRateScheduler(step_decay(1.0, drop=0.5, every=2))
        model.fit(X, y, epochs=5, batch_size=4,
                  callbacks=[scheduler], verbose=0)
        # After 5 epochs (indices 0..4), the scale has halved twice.
        self.assertAlmostEqual(model.optimizer.lr_scale, 0.25)

    def test_cosine_decay_shape(self):
        schedule = cosine_decay(1.0, total_epochs=10, min_scale=0.0)
        self.assertAlmostEqual(schedule(0), 1.0)
        self.assertAlmostEqual(schedule(10), 0.0, places=10)
        self.assertAlmostEqual(schedule(5), 0.5, places=10)


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------

class TestDataPipeline(unittest.TestCase):

    def test_train_test_split_sizes_and_disjointness(self):
        X = [[float(i)] for i in range(100)]
        y = [[float(i % 2)] for i in range(100)]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        train_values = {row[0] for row in X_tr}
        test_values = {row[0] for row in X_te}
        self.assertEqual(train_values & test_values, set())

    def test_split_keeps_x_and_y_paired(self):
        X = [[float(i)] for i in range(50)]
        y = [[float(i)] for i in range(50)]   # y mirrors X, so pairing is checkable
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
        for xi, yi in zip(X_tr + X_te, y_tr + y_te):
            self.assertEqual(xi[0], yi[0])

    def test_stratified_split_preserves_class_balance(self):
        X = [[float(i)] for i in range(100)]
        y = [0] * 90 + [1] * 10          # 10% minority class
        _, _, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=True)
        self.assertGreater(sum(1 for v in y_te if v == 1), 0)

    def test_three_way_split(self):
        X = [[float(i)] for i in range(100)]
        y = [[float(i % 2)] for i in range(100)]
        X_tr, X_v, X_te, *_ = train_val_test_split(X, y, val_size=0.2, test_size=0.2)
        self.assertEqual(len(X_tr) + len(X_v) + len(X_te), 100)
        self.assertAlmostEqual(len(X_v) / 100, 0.2, delta=0.02)

    def test_split_rejects_impossible_sizes(self):
        X, y = [[1.0]] * 10, [[0.0]] * 10
        with self.assertRaises(ValueError):
            train_val_test_split(X, y, val_size=0.6, test_size=0.6)

    def test_standard_scaler_produces_zero_mean_unit_std(self):
        X = [[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]]
        scaled = StandardScaler().fit_transform(X)
        for col in range(2):
            values = [row[col] for row in scaled]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            self.assertAlmostEqual(mean, 0.0, places=10)
            self.assertAlmostEqual(var, 1.0, places=10)

    def test_standard_scaler_inverse(self):
        X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        scaler = StandardScaler().fit(X)
        restored = scaler.inverse_transform(scaler.transform(X))
        for original, back in zip(X, restored):
            for a, b in zip(original, back):
                self.assertAlmostEqual(a, b, places=10)

    def test_scaler_handles_constant_column(self):
        scaled = StandardScaler().fit_transform([[5.0], [5.0], [5.0]])
        self.assertTrue(all(row[0] == 0.0 for row in scaled))

    def test_minmax_scaler_range(self):
        X = [[1.0], [5.0], [10.0]]
        scaled = MinMaxScaler().fit_transform(X)
        self.assertAlmostEqual(scaled[0][0], 0.0)
        self.assertAlmostEqual(scaled[2][0], 1.0)

    def test_minmax_custom_range(self):
        scaled = MinMaxScaler((-1.0, 1.0)).fit_transform([[0.0], [10.0]])
        self.assertAlmostEqual(scaled[0][0], -1.0)
        self.assertAlmostEqual(scaled[1][0], 1.0)

    def test_transform_before_fit_is_an_error(self):
        # This is the guard against data leakage.
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1.0]])

    def test_scaler_uses_only_training_statistics(self):
        train = [[0.0], [10.0]]
        scaler = StandardScaler().fit(train)
        mean_before = list(scaler.mean)
        scaler.transform([[1000.0]])       # transforming must not refit
        self.assertEqual(scaler.mean, mean_before)

    def test_wrong_feature_count_rejected(self):
        scaler = StandardScaler().fit([[1.0, 2.0]])
        with self.assertRaises(Exception):
            scaler.transform([[1.0]])

    def test_label_encoder(self):
        encoder = LabelEncoder()
        indices = encoder.fit_transform(["cat", "dog", "cat", "bird"])
        self.assertEqual(encoder.num_classes, 3)
        self.assertEqual(len(indices), 4)
        self.assertEqual(indices[0], indices[2])
        self.assertEqual(encoder.inverse_transform(indices)[0], "cat")

    def test_label_encoder_is_deterministic(self):
        a = LabelEncoder().fit(["b", "a", "c"]).classes
        b = LabelEncoder().fit(["c", "b", "a"]).classes
        self.assertEqual(a, b)

    def test_label_encoder_rejects_unseen(self):
        encoder = LabelEncoder().fit(["a", "b"])
        with self.assertRaises(ValueError):
            encoder.transform(["c"])

    def test_one_hot(self):
        self.assertEqual(one_hot([0, 2, 1], 3),
                         [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def test_one_hot_roundtrip(self):
        labels = [0, 2, 1, 3]
        self.assertEqual(from_one_hot(one_hot(labels, 4)), labels)

    def test_one_hot_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            one_hot([5], 3)

    def test_batch_iterator_covers_every_sample(self):
        X = [[float(i)] for i in range(10)]
        y = [[float(i)] for i in range(10)]
        seen = []
        for xb, yb in batch_iterator(X, y, batch_size=3, shuffle=False):
            seen.extend(xb.data)
        self.assertEqual(sorted(seen), [float(i) for i in range(10)])

    def test_batch_iterator_final_partial_batch(self):
        X = [[float(i)] for i in range(10)]
        sizes = [xb.shape[0] for xb, _ in batch_iterator(X, X, 3, shuffle=False)]
        self.assertEqual(sizes, [3, 3, 3, 1])

    def test_batch_iterator_drop_last(self):
        X = [[float(i)] for i in range(10)]
        sizes = [xb.shape[0] for xb, _ in
                 batch_iterator(X, X, 3, shuffle=False, drop_last=True)]
        self.assertEqual(sizes, [3, 3, 3])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            save_csv(path, [[1.0, 2.0, "a"], [3.0, 4.0, "b"]],
                     header=["x1", "x2", "label"])
            X, y, names = load_csv(path, target_column=-1)
            self.assertEqual(X, [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(y, ["a", "b"])
            self.assertEqual(names, ["x1", "x2"])

    def test_csv_target_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            save_csv(path, [[1.0, 9.0, 2.0]], header=["a", "target", "b"])
            X, y, names = load_csv(path, target_column="target")
            self.assertEqual(X, [[1.0, 2.0]])
            self.assertEqual(y, [9.0])

    def test_csv_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            save_csv(path, [[1.0, "", 0], [3.0, 5.0, 1]], header=["a", "b", "y"])
            with self.assertRaises(ValueError):
                load_csv(path)
            X, _, _ = load_csv(path, missing="zero")
            self.assertEqual(X[0][1], 0.0)
            X, _, _ = load_csv(path, missing="mean")
            self.assertEqual(X[0][1], 5.0)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        self.assertEqual(accuracy([[0.9], [0.1], [0.8]], [[1.0], [0.0], [1.0]]), 1.0)
        self.assertEqual(accuracy([[0.9], [0.9]], [[1.0], [0.0]]), 0.5)

    def test_confusion_matrix_layout(self):
        # rows = actual, columns = predicted
        matrix = confusion_matrix([[0.9], [0.1], [0.9], [0.1]],
                                  [[1.0], [0.0], [0.0], [1.0]])
        self.assertEqual(matrix[1][1], 1)   # true positive
        self.assertEqual(matrix[0][0], 1)   # true negative
        self.assertEqual(matrix[0][1], 1)   # false positive
        self.assertEqual(matrix[1][0], 1)   # false negative

    def test_precision_and_recall(self):
        # 2 predicted positive, 1 correct -> precision 0.5
        # 2 actual positive, 1 found     -> recall 0.5
        pred = [[0.9], [0.9], [0.1], [0.1]]
        true = [[1.0], [0.0], [1.0], [0.0]]
        self.assertAlmostEqual(precision(pred, true), 0.5)
        self.assertAlmostEqual(recall(pred, true), 0.5)

    def test_f1_is_the_harmonic_mean(self):
        pred = [[0.9], [0.9], [0.1], [0.1]]
        true = [[1.0], [0.0], [1.0], [0.0]]
        self.assertAlmostEqual(f1_score(pred, true), 0.5)

    def test_f1_punishes_an_imbalanced_pair(self):
        # Predicting everything positive: recall 1.0, precision 0.5.
        # The harmonic mean (0.667) is below the arithmetic mean (0.75).
        pred = [[0.9]] * 4
        true = [[1.0], [1.0], [0.0], [0.0]]
        self.assertAlmostEqual(f1_score(pred, true), 2 * 0.5 * 1.0 / 1.5)

    def test_multiclass_accuracy_uses_argmax(self):
        pred = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
        true = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.assertEqual(accuracy(pred, true), 1.0)

    def test_regression_metrics(self):
        pred = [[2.0], [4.0]]
        true = [[1.0], [3.0]]
        self.assertAlmostEqual(mean_absolute_error(pred, true), 1.0)
        self.assertAlmostEqual(root_mean_squared_error(pred, true), 1.0)

    def test_r_squared(self):
        # Perfect predictions -> 1.0
        self.assertAlmostEqual(r_squared([[1.0], [2.0], [3.0]],
                                         [[1.0], [2.0], [3.0]]), 1.0)
        # Always predicting the mean -> 0.0
        self.assertAlmostEqual(r_squared([[2.0], [2.0], [2.0]],
                                         [[1.0], [2.0], [3.0]]), 0.0)

    def test_r_squared_can_be_negative(self):
        self.assertLess(r_squared([[10.0], [10.0], [10.0]],
                                  [[1.0], [2.0], [3.0]]), 0.0)


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def _trained_model(self):
        set_seed(5)
        X, y = xor_dataset()
        model = Sequential([Dense(2, 6, activation="tanh"),
                            Dense(6, 1, activation="sigmoid")])
        model.compile(loss="binary_cross_entropy", optimizer=Adam(0.05))
        model.fit(X, y, epochs=100, batch_size=4, verbose=0)
        return model, X

    def test_predictions_are_identical_after_roundtrip(self):
        model, X = self._trained_model()
        before = model.predict(X).data
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            reloaded = Sequential.load(path)
            after = reloaded.predict(X).data
        # Exact equality: repr(float) round-trips losslessly through JSON.
        self.assertEqual(before, after)

    def test_saved_file_is_readable_json(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        self.assertEqual(payload["format"], "scratch_nn")
        self.assertIn("architecture", payload)
        self.assertIn("weights", payload)

    def test_architecture_is_restored(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            reloaded = Sequential.load(path)
        self.assertEqual(len(reloaded.layers), len(model.layers))
        self.assertEqual(reloaded.layers[0].input_size, 2)
        self.assertEqual(reloaded.layers[0].activation_name, "tanh")

    def test_optimizer_state_is_restored(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            reloaded = Sequential.load(path)
        self.assertEqual(reloaded.optimizer.iterations, model.optimizer.iterations)
        self.assertIsInstance(reloaded.optimizer, Adam)

    def test_training_resumes_after_loading(self):
        model, X = self._trained_model()
        _, y = xor_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            reloaded = Sequential.load(path)
            history = reloaded.fit(X, y, epochs=10, batch_size=4, verbose=0)
        self.assertEqual(len(history["loss"]), 10)

    def test_scaler_and_encoder_are_persisted(self):
        model, _ = self._trained_model()
        scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
        encoder = LabelEncoder().fit(["a", "b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            save_model(model, path, scaler=scaler, encoder=encoder)
            _, extras = load_model(path, with_extras=True)
        self.assertIn("scaler", extras)
        self.assertEqual(extras["scaler"].mean, scaler.mean)
        self.assertEqual(extras["encoder"].classes, ["a", "b"])

    def test_metadata_is_persisted(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            save_model(model, path, metadata={"note": "hello"})
            _, extras = load_model(path, with_extras=True)
        self.assertEqual(extras["metadata"]["note"], "hello")

    def test_dropout_layer_survives_roundtrip(self):
        set_seed(6)
        model = Sequential([Dense(2, 4, activation="relu"),
                            Dropout(0.3),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            reloaded = Sequential.load(path)
        self.assertIsInstance(reloaded.layers[1], Dropout)
        self.assertAlmostEqual(reloaded.layers[1].rate, 0.3)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_model("does_not_exist_12345.json")

    def test_foreign_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "other.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"hello": "world"}, fh)
            with self.assertRaises(ValueError):
                load_model(path)

    def test_diverged_model_is_not_written(self):
        model = Sequential([Dense(2, 2, activation="identity")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        model.layers[0].W.data[0] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with self.assertRaises(ValueError):
                model.save(path)
            self.assertFalse(os.path.exists(path))


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

class TestInference(unittest.TestCase):

    def test_predict_accepts_a_single_sample(self):
        set_seed(8)
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        self.assertEqual(model.predict([[0, 1]]).shape, (1, 1))

    def test_predict_classes_binary(self):
        set_seed(9)
        X, y = xor_dataset()
        model = Sequential([Dense(2, 8, activation="tanh"),
                            Dense(8, 1, activation="sigmoid")])
        model.compile(loss="binary_cross_entropy", optimizer=Adam(0.05))
        model.fit(X, y, epochs=400, batch_size=4, verbose=0)
        self.assertEqual(model.predict_classes(X), [0, 1, 1, 0])

    def test_predict_proba_binary_returns_two_columns(self):
        set_seed(10)
        model = Sequential([Dense(2, 4, activation="tanh"),
                            Dense(4, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        probs = model.predict_proba([[0, 1]])
        self.assertEqual(len(probs[0]), 2)
        self.assertAlmostEqual(sum(probs[0]), 1.0, places=10)

    def test_dropout_is_inactive_at_inference(self):
        set_seed(12)
        model = Sequential([Dense(2, 20, activation="relu"),
                            Dropout(0.5),
                            Dense(20, 1, activation="sigmoid")])
        model.compile(loss="mse", optimizer=SGD(0.1))
        first = model.predict([[0.5, 0.5]]).data[0]
        second = model.predict([[0.5, 0.5]]).data[0]
        self.assertEqual(first, second)

    def test_multiclass_probabilities_sum_to_one(self):
        set_seed(13)
        model = Sequential([Dense(2, 5, activation="tanh"),
                            Dense(5, 3, activation="softmax")])
        model.compile(loss="categorical_cross_entropy", optimizer=Adam(0.01))
        for row in model.predict_proba([[0.1, 0.2], [1.0, -1.0]]):
            self.assertAlmostEqual(sum(row), 1.0, places=10)


class TestF1Score(unittest.TestCase):
    """F1 is the harmonic mean of precision and recall."""

    def test_perfect_classifier_scores_one(self):
        self.assertEqual(f1_score([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_matches_hand_computed_value(self):
        # predicted 1s: indices 0,1,2 -> TP=2 (0,2), FP=1 (1)
        # actual 1s:    indices 0,2,3 -> FN=1 (3)
        y_pred = [1, 1, 1, 0]
        y_true = [1, 0, 1, 1]
        p = 2 / 3
        r = 2 / 3
        self.assertAlmostEqual(precision(y_pred, y_true), p, places=12)
        self.assertAlmostEqual(recall(y_pred, y_true), r, places=12)
        self.assertAlmostEqual(f1_score(y_pred, y_true),
                               2 * p * r / (p + r), places=12)

    def test_harmonic_mean_punishes_imbalance(self):
        # Predicting every sample positive gives recall 1 but poor precision.
        # The arithmetic mean would flatter it; the harmonic mean must not.
        y_pred = [1, 1, 1, 1]
        y_true = [1, 0, 0, 0]
        self.assertEqual(recall(y_pred, y_true), 1.0)
        self.assertAlmostEqual(precision(y_pred, y_true), 0.25, places=12)
        f1 = f1_score(y_pred, y_true)
        self.assertAlmostEqual(f1, 0.4, places=12)
        self.assertLess(f1, (1.0 + 0.25) / 2)

    def test_no_predicted_positives_is_zero_not_error(self):
        self.assertEqual(f1_score([0, 0, 0], [1, 0, 1]), 0.0)

    def test_macro_average_treats_classes_equally(self):
        # Class 0 is perfect, class 1 is never predicted -> (1 + 0) / 2.
        y_pred = [0, 0, 0, 0]
        y_true = [0, 0, 0, 1]
        self.assertAlmostEqual(f1_score(y_pred, y_true, average="macro"),
                               (2 * (3 / 4) * 1.0 / ((3 / 4) + 1.0) + 0.0) / 2,
                               places=12)

    def test_agrees_with_classification_report(self):
        y_pred = [1, 1, 0, 1, 0, 0]
        y_true = [1, 0, 0, 1, 1, 0]
        report = classification_report(y_pred, y_true)
        self.assertIn(f"{f1_score(y_pred, y_true):.4f}", report)


class TestConfusionMatrixSVG(unittest.TestCase):
    """The confusion graph: an SVG heatmap of the confusion matrix."""

    MATRIX = [[47, 2, 1], [3, 51, 0], [0, 4, 44]]

    def test_output_is_well_formed_xml(self):
        svg = confusion_matrix_svg(self.MATRIX)
        xml.dom.minidom.parseString(svg)  # raises on malformed markup
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.rstrip().endswith("</svg>"))

    def test_every_count_appears_in_the_output(self):
        svg = confusion_matrix_svg(self.MATRIX, normalize=False)
        for row in self.MATRIX:
            for count in row:
                self.assertIn(f">{count}</text>", svg)

    def test_class_names_are_rendered(self):
        svg = confusion_matrix_svg([[1, 0], [0, 1]], class_names=["cat", "dog"])
        self.assertIn("cat", svg)
        self.assertIn("dog", svg)

    def test_class_names_are_xml_escaped(self):
        # A name containing markup must not be able to break the document.
        svg = confusion_matrix_svg([[1, 0], [0, 1]],
                                   class_names=["<script>", "a&b"])
        xml.dom.minidom.parseString(svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("a&amp;b", svg)
        self.assertNotIn("<script>", svg)

    def test_diagonal_and_off_diagonal_use_different_hues(self):
        # Correct cells are green, errors red, so the two must not collide.
        svg = confusion_matrix_svg([[10, 0], [0, 10]])
        other = confusion_matrix_svg([[0, 10], [10, 0]])
        self.assertNotEqual(svg, other)

    def test_empty_rows_do_not_divide_by_zero(self):
        svg = confusion_matrix_svg([[0, 0], [0, 0]])
        xml.dom.minidom.parseString(svg)

    def test_non_square_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            confusion_matrix_svg([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            confusion_matrix_svg([])

    def test_accepts_output_of_confusion_matrix(self):
        y_pred = [1, 0, 1, 1, 0]
        y_true = [1, 0, 0, 1, 0]
        svg = confusion_matrix_svg(confusion_matrix(y_pred, y_true))
        xml.dom.minidom.parseString(svg)

    def test_save_writes_a_file_and_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reports", "cm.svg")
            returned = save_confusion_matrix_svg(self.MATRIX, path)
            self.assertEqual(returned, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                xml.dom.minidom.parseString(fh.read())


class TestPickleSerialization(unittest.TestCase):
    """.pkl checkpoints carry the same payload as the JSON ones."""

    def _trained_model(self):
        set_seed(5)
        X, y = xor_dataset()
        model = Sequential([Dense(2, 6, activation="tanh"),
                            Dense(6, 1, activation="sigmoid")])
        model.compile(loss="binary_cross_entropy", optimizer=Adam(0.05))
        model.fit(X, y, epochs=100, batch_size=4, verbose=0)
        return model, X

    def test_predictions_are_identical_after_roundtrip(self):
        model, X = self._trained_model()
        before = model.predict(X).data
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            model.save_pickle(path)
            after = Sequential.load_pickle(path).predict(X).data
        self.assertEqual(before, after)

    def test_json_and_pickle_agree(self):
        model, X = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            jpath = os.path.join(tmp, "m.json")
            ppath = os.path.join(tmp, "m.pkl")
            model.save(jpath)
            model.save_pickle(ppath)
            from_json = Sequential.load(jpath).predict(X).data
            from_pickle = Sequential.load_pickle(ppath).predict(X).data
        self.assertEqual(from_json, from_pickle)

    def test_payload_is_a_plain_dict_not_a_live_model(self):
        # Storing the dictionary rather than the object keeps loading routed
        # through model_from_dict.
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            model.save_pickle(path)
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["format"], "scratch_nn")
        self.assertIn("weights", payload)

    def test_architecture_is_restored(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            model.save_pickle(path)
            reloaded = Sequential.load_pickle(path)
        self.assertEqual(len(reloaded.layers), len(model.layers))
        self.assertEqual(reloaded.layers[0].input_size, 2)
        self.assertEqual(reloaded.layers[0].activation_name, "tanh")

    def test_creates_parent_directories(self):
        model, _ = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "models", "nested", "m.pkl")
            model.save_pickle(path)
            self.assertTrue(os.path.exists(path))

    def test_non_finite_weights_are_rejected(self):
        model = Sequential([Dense(2, 2, activation="relu")])
        model.layers[0].W.data[0] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            with self.assertRaises(ValueError):
                model.save_pickle(path)
            # A failed save must leave no half-written file behind.
            self.assertEqual(os.listdir(tmp), [])

    def test_missing_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_pickle(os.path.join(tmp, "nope.pkl"))

    def test_unreadable_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "garbage.pkl")
            with open(path, "wb") as fh:
                fh.write(b"this is not a pickle")
            with self.assertRaises(ValueError):
                load_pickle(path)

    def test_pickle_of_wrong_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrong.pkl")
            with open(path, "wb") as fh:
                pickle.dump([1, 2, 3], fh)
            with self.assertRaises(ValueError):
                load_pickle(path)

    def test_metadata_survives_the_roundtrip(self):
        model, X = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            save_pickle(model, path, metadata={"note": "xor"})
            _, extras = load_pickle(path, with_extras=True)
        self.assertEqual(extras["metadata"]["note"], "xor")


if __name__ == "__main__":
    unittest.main()

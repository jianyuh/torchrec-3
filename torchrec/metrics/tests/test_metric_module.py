#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import os
import tempfile
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.metric_module import (
    generate_metric_module,
    MetricValue,
    RecMetricModule,
    StateMetric,
    StateMetricEnum,
)
from torchrec.metrics.metrics_config import (
    _DEFAULT_WINDOW_SIZE,
    DefaultMetricsConfig,
    DefaultTaskInfo,
    EmptyMetricsConfig,
    RecMetricDef,
    RecMetricEnum,
)
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecMetricList, RecTaskInfo
from torchrec.metrics.tests.test_utils import gen_test_batch, get_launch_config
from torchrec.metrics.throughput import ThroughputMetric

METRIC_MODULE_PATH = "torchrec.metrics.metric_module"


class MockOptimizer(StateMetric):
    def __init__(self) -> None:
        self.get_metrics_call = 0

    def get_metrics(self) -> Dict[str, MetricValue]:
        self.get_metrics_call += 1
        return {"learning_rate": torch.tensor(1.0)}


class TestMetricModule(RecMetricModule):
    r"""Implementation of RecMetricModule."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        rec_tasks: Optional[List[RecTaskInfo]] = None,
        rec_metrics: Optional[RecMetricList] = None,
        throughput_metric: Optional[ThroughputMetric] = None,
        state_metrics: Optional[Dict[str, StateMetric]] = None,
        compute_interval_steps: int = 100,
        memory_usage_limit_mb: float = 512,
    ) -> None:
        super().__init__(
            batch_size,
            world_size,
            rec_tasks=rec_tasks,
            rec_metrics=rec_metrics,
            throughput_metric=throughput_metric,
            state_metrics=state_metrics,
            compute_interval_steps=compute_interval_steps,
            memory_usage_limit_mb=memory_usage_limit_mb,
        )

    def _update_rec_metrics(self, model_out: Dict[str, torch.Tensor]) -> None:
        if isinstance(model_out, MagicMock):
            return
        labels, predictions, weights = parse_task_model_outputs(
            self.rec_tasks, model_out
        )
        self.rec_metrics.update(predictions=predictions, labels=labels, weights=weights)


class MetricModuleTest(unittest.TestCase):
    def test_metric_module(self) -> None:
        rec_metric_list_patch = patch(
            METRIC_MODULE_PATH + ".RecMetricList",
        )

        with tempfile.NamedTemporaryFile(delete=True) as backend:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend.name}",
                world_size=1,
                rank=0,
            )
            for pg in [None, dist.new_group([0])]:
                rec_metric_list_mock = rec_metric_list_patch.start()
                mock_optimizer = MockOptimizer()
                config = dataclasses.replace(
                    DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
                )
                metric_module = generate_metric_module(
                    TestMetricModule,
                    metrics_config=config,
                    batch_size=128,
                    world_size=64,
                    my_rank=0,
                    state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
                    device=torch.device("cpu"),
                    process_group=pg,
                )
                metric_module.rec_metrics.compute = MagicMock(
                    return_value={"ne-ne|lifetime_ne": torch.tensor(0.75)}
                )
                self.assertEqual(
                    len(rec_metric_list_mock.call_args[0][0]),
                    len(DefaultMetricsConfig.rec_metrics),
                )
                self.assertEqual(len(metric_module.state_metrics), 1)
                metric_module.update(MagicMock())
                ret = metric_module.compute()
                rec_metric_list_patch.stop()
                metric_module.rec_metrics.compute.assert_called_once()
                self.assertTrue("ne-ne|lifetime_ne" in ret)
                self.assertTrue("throughput-throughput|total_examples" in ret)
                self.assertTrue("optimizers-optimizers|learning_rate" in ret)
            dist.destroy_process_group()

    def test_rectask_info(self) -> None:
        mock_optimizer = MockOptimizer()
        config = DefaultMetricsConfig
        metric_module_seperate_task_info = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.NE: RecMetricDef(
                    rec_tasks=[], rec_task_indices=[0], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
        )
        metric_module_unified_task_info = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        self.assertEqual(
            metric_module_seperate_task_info.rec_metrics[0]._namespace,
            metric_module_unified_task_info.rec_metrics[0]._namespace,
        )
        self.assertEqual(
            metric_module_seperate_task_info.rec_metrics[0]._tasks,
            metric_module_unified_task_info.rec_metrics[0]._tasks,
        )

    @staticmethod
    def _run_trainer_checkpointing() -> None:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
        )
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=world_size,
            my_rank=rank,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        value = 12345
        state_dict = metric_module.state_dict()
        keys = list(state_dict.keys())
        for k in state_dict.keys():
            state_dict[k] = torch.tensor(value, dtype=torch.long).detach()
        logging.info(f"Metrics state keys = {keys}")
        metric_module.load_state_dict(state_dict)
        tc = unittest.TestCase()
        tc.assertTrue("throughput_metric.warmup_examples" in keys)
        tc.assertTrue("throughput_metric.total_examples" in keys)
        tc.assertTrue(
            "rec_metrics.rec_metrics.0._metrics_computations.0.cross_entropy_sum"
            in keys
        )

        # 1. Test sync()
        metric_module.sync()
        state_dict = metric_module.state_dict()
        for k, v in state_dict.items():
            if k.startswith("rec_metrics."):
                if k.endswith("has_valid_update"):
                    tc.assertEqual(v.item(), 1)
                else:
                    tc.assertEqual(v.item(), value * world_size)

        # 2. Test unsync()
        metric_module.unsync()
        state_dict = metric_module.state_dict()
        for v in state_dict.values():
            tc.assertEqual(v.item(), value)

        # 3. Test reset()
        metric_module.reset()
        state_dict = metric_module.state_dict()
        for k, v in state_dict.items():
            if k.startswith("rec_metrics."):
                tc.assertEqual(v.item(), 0)

    def test_rank0_checkpointing(self) -> None:
        # Call the tested methods to make code coverage visible to the testing system
        # Begin of dummy codes
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        state_dict = metric_module.state_dict()
        metric_module.load_state_dict(state_dict)
        metric_module.sync()
        metric_module.unsync()
        metric_module.reset()
        # End of dummy codes

        with tempfile.TemporaryDirectory() as tmpdir:
            lc = get_launch_config(
                world_size=2, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
            )
            pet.elastic_launch(lc, entrypoint=self._run_trainer_checkpointing)()

    @staticmethod
    def _run_trainer_initial_states_checkpointing() -> None:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.AUC: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
            state_metrics=[StateMetricEnum.OPTIMIZERS],
        )

        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(metric_module.rec_metrics.rec_metrics[0], AUCMetric))
        tc.assertEqual(
            len(
                metric_module.rec_metrics.rec_metrics[0]
                ._metrics_computations[0]
                .predictions
            ),
            1,  # The predictions state is a list containing 1 tensor value
        )

        # 1. After the metric module is created
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
            # The 1st 1 is the number of tasks; the 2nd 1 is the default value length
        )

        metric_module.sync()
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 2),
        )

        metric_module.unsync()
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
        )

        # 2. After the metric module gets reset
        metric_module.update(gen_test_batch(128))
        metric_module.reset()
        metric_module.sync()
        metric_module.unsync()
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
        )

    def test_initial_states_rank0_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = get_launch_config(
                world_size=2, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
            )
            pet.elastic_launch(
                lc, entrypoint=self._run_trainer_initial_states_checkpointing
            )()

    def test_empty_memory_usage(self) -> None:
        mock_optimizer = MockOptimizer()
        config = EmptyMetricsConfig
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        self.assertEqual(metric_module.get_memory_usage(), 0)

    def test_ne_memory_usage(self) -> None:
        mock_optimizer = MockOptimizer()
        config = DefaultMetricsConfig
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        # Default NEMetric's dtype is
        #   float64 (8 bytes) * 16 tensors of size 1 + unit8 (1 byte) * 2 tensors of size 1 = 130 bytes
        #       Tensors in NeMetricComputation:
        #           NE metric specific attributes: 8 in _default, 8 actual attribute values: 4 attributes, 4 window
        #           RecMetric's has_valid_update attribute: 1 in _default, 1 actual attribute value
        self.assertEqual(metric_module.get_memory_usage(), 130)
        metric_module.update(gen_test_batch(128))
        self.assertEqual(metric_module.get_memory_usage(), 162)

    def test_calibration_memory_usage(self) -> None:
        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.CALIBRATION: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
        )
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        # Default calibration metric dtype is
        #   float64 (8 bytes) * 8 tensors of size 1 + uint8 (1 byte) * 2 tensors of size 1 = 66 bytes
        #       Tensors in CalibrationMetricComputation:
        #           Calibration metric attributes: 4 in _default, 4 actual attribute values: 2 attribute, 2 window
        #           RecMetric's has_valid_update attribute: 1 in _default, 1 actual attribute value
        self.assertEqual(metric_module.get_memory_usage(), 66)
        metric_module.update(gen_test_batch(128))
        self.assertEqual(metric_module.get_memory_usage(), 82)

    def test_auc_memory_usage(self) -> None:
        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.AUC: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
        )
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        # 3 (tensors) * 8 (double) + 1 (tensor) * 2 (uint8)
        self.assertEqual(metric_module.get_memory_usage(), 26)
        metric_module.update(gen_test_batch(128))
        # 24 (initial states) + 3 (tensors) * 128 (batch_size) * 8 (double) + 1 (tensor) * 2 (uint8)
        self.assertEqual(metric_module.get_memory_usage(), 3098)

    def test_check_memory_usage(self) -> None:
        mock_optimizer = MockOptimizer()
        config = DefaultMetricsConfig
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        metric_module.update(gen_test_batch(128))
        with patch("torchrec.metrics.metric_module.logger") as logger_mock:
            # Memory usage is fine.
            metric_module.memory_usage_mb_avg = 160 / (10 ** 6)
            metric_module.check_memory_usage(1000)
            self.assertEqual(metric_module.oom_count, 0)
            self.assertEqual(logger_mock.warning.call_count, 0)

            # OOM but memory usage does not exceed avg.
            metric_module.memory_usage_limit_mb = 0.000001
            metric_module.memory_usage_mb_avg = 160 / (10 ** 6)
            metric_module.check_memory_usage(1000)
            self.assertEqual(metric_module.oom_count, 1)
            self.assertEqual(logger_mock.warning.call_count, 1)

            # OOM and memory usage exceed avg but warmup is not over.
            metric_module.memory_usage_mb_avg = 160 / (10 ** 6) / 10
            metric_module.check_memory_usage(2)
            self.assertEqual(metric_module.oom_count, 2)
            self.assertEqual(logger_mock.warning.call_count, 2)

            # OOM and memory usage exceed avg and warmup is over.
            metric_module.memory_usage_mb_avg = 160 / (10 ** 6) / 1.25
            metric_module.check_memory_usage(1002)
            self.assertEqual(metric_module.oom_count, 3)
            self.assertEqual(logger_mock.warning.call_count, 4)

    def test_should_compute(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        metric_module.trained_batches = 1
        self.assertFalse(metric_module.should_compute())
        metric_module.trained_batches = metric_module.compute_interval_steps - 1
        self.assertFalse(metric_module.should_compute())
        metric_module.trained_batches = metric_module.compute_interval_steps
        self.assertTrue(metric_module.should_compute())

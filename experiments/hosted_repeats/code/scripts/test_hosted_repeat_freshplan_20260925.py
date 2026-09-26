#!/usr/bin/env python3
"""CPU-only regression using the frozen planner function and actual wrapper."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import run_hosted_repeat40_freshplan_20260925 as runner


class FreshPlannerTest(unittest.TestCase):
    def setUp(self):
        source = runner.SOURCE / 'agents/agent_vlm.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'make_episode_plan')
        self.calls = []
        def client(*args, **kwargs):
            self.calls.append((args, kwargs))
            return json.dumps({'target_guess': 'sink', 'candidate_objects': ['sink'],
                               'action_plan': ['find bathroom']}), None, {'calls': 1}
        namespace = {'_plan_cache': {}, 'PLAN_SYSTEM': 'test prompt', 'call_vlm': client,
                     'tolerant_json_parse': lambda value: (json.loads(value), None)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        class PlannerProxy:
            def __getattr__(self, key):
                return namespace[key]
            def __setattr__(self, key, value):
                namespace[key] = value
        self.planner = PlannerProxy()
        self.agent = SimpleNamespace(make_episode_plan=self.planner.make_episode_plan,
                                     _scene_plan_cache={'stale': {'target_guess': 'table'}})
        self.args = ('wash my hands', 'test-model')
        self.kwargs = {'sel_id': 'SEL_001', 'style': 'formal'}

    def test_repeated_key_reproduces_bug_then_each_episode_calls_client(self):
        self.agent.make_episode_plan(*self.args, **self.kwargs)
        cached, usage = self.agent.make_episode_plan(*self.args, **self.kwargs)
        self.assertTrue(cached['_cached'])
        self.assertIsNone(usage)
        self.assertEqual(len(self.calls), 1)
        for i in range(3):
            with runner.fresh_episode_plan(self.agent, self.planner) as witnesses:
                plan, usage = self.agent.make_episode_plan(*self.args, **self.kwargs)
                self.assertEqual(self.agent._scene_plan_cache, {})
                self.assertNotIn('_cached', plan)
                self.assertEqual(witnesses[0]['plan_sha256'], runner.json_digest(plan))
            self.assertEqual(len(witnesses), 1)
            self.assertEqual(witnesses[0]['client_calls'], 1)
            self.assertEqual(len(self.calls), i + 2)
            self.assertEqual(self.planner._plan_cache, {})
            self.assertIs(self.agent.make_episode_plan, self.planner.make_episode_plan)

    def test_missing_planner_call_aborts(self):
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            with runner.fresh_episode_plan(self.agent, self.planner):
                pass
        self.assertIs(self.agent.make_episode_plan, self.planner.make_episode_plan)

    def test_serialized_plan_metadata_and_tampering(self):
        with runner.fresh_episode_plan(self.agent, self.planner) as witnesses:
            plan, usage = self.agent.make_episode_plan(*self.args, **self.kwargs)
        # Frozen engine serializes {**plan, plan_model: model, plan_usage: usage}.
        recorded = {**plan, 'plan_model': 'test-model', 'plan_usage': usage}
        runner.verify_record_plan(recorded, witnesses[0], 'test-model')
        with self.assertRaisesRegex(RuntimeError, 'content differs'):
            runner.verify_record_plan({**recorded, 'target_guess': 'table'}, witnesses[0], 'test-model')
        with self.assertRaisesRegex(RuntimeError, 'metadata differs'):
            runner.verify_record_plan({**recorded, 'plan_usage': None}, witnesses[0], 'test-model')
        with self.assertRaisesRegex(RuntimeError, 'metadata differs'):
            runner.verify_record_plan(recorded, witnesses[0], 'different-model')

    def test_cache_hit_aborts_even_with_a_witness_wrapper(self):
        with self.assertRaisesRegex(RuntimeError, 'not witnessed'):
            with runner.fresh_episode_plan(self.agent, self.planner):
                self.planner._plan_cache[('SEL_001', 'formal', 'test-model')] = {'target_guess': 'sink'}
                self.agent.make_episode_plan(*self.args, **self.kwargs)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.planner._plan_cache, {})


if __name__ == '__main__':
    unittest.main()

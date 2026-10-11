"""A2 configuration and wiring checks without torch imports or model execution."""
import argparse
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parent
TREE = ast.parse((ROOT / "main_restoration.py").read_text(encoding="utf-8-sig"))
HELPERS = {"get_args_parser", "_saved_argument", "_resolve_degradation_mode", "_resolve_psf_settings"}


class A2EntryTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"argparse": argparse, "Path": Path, "math": math,
                   "PROJECT_ROOT": ROOT.parent, "JIT_ROOT": ROOT}
        nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "a2-entry-only", "exec"), self.ns)

    def resolve(self, mode="psf_local", weight=None, eps=None, checkpoint=None):
        args = SimpleNamespace(degradation_mode=mode, lambda_psf=weight, psf_eps=eps)
        self.ns["_resolve_psf_settings"](args, checkpoint)
        return args

    def test_explicit_a2_weight_and_zero_ablation(self):
        parser = self.ns["get_args_parser"]()
        args = parser.parse_args(["--degradation_mode", "psf_local", "--lambda_psf", ".1"])
        self.ns["_resolve_psf_settings"](args)
        self.assertEqual((args.lambda_psf, args.psf_eps), (.1, .001))
        self.assertEqual(self.resolve(weight=0).lambda_psf, 0)
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.resolve()

    def test_legacy_and_a1_objectives_remain_zero(self):
        for mode in ("none", "fixed", "predicted", "fixed_local", "predicted_local"):
            self.assertEqual(self.resolve(mode=mode).lambda_psf, 0)
            with self.assertRaises(ValueError):
                self.resolve(mode=mode, weight=.1)
        self.assertEqual(self.resolve(mode="none", checkpoint={"args": {}}).lambda_psf, 0)

    def test_resume_inherits_and_rejects_objective_changes(self):
        checkpoint = {"args": {"degradation_mode": "psf_local", "lambda_psf": .1, "psf_eps": .002}}
        self.assertEqual(self.ns["_resolve_degradation_mode"](None, checkpoint), "psf_local")
        args = self.resolve(checkpoint=checkpoint)
        self.assertEqual((args.lambda_psf, args.psf_eps), (.1, .002))
        for weight, eps in ((.2, None), (None, .003)):
            with self.assertRaisesRegex(ValueError, "Resume"):
                self.resolve(weight=weight, eps=eps, checkpoint=checkpoint)
        with self.assertRaises(ValueError):
            self.ns["_resolve_degradation_mode"]("predicted_local", checkpoint)

    def test_numeric_parameter_validation(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.resolve(weight=value)
        for eps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.resolve(weight=.1, eps=eps)

    def test_a2_does_not_construct_condition_adapters(self):
        tree = ast.parse((ROOT / "denoiser.py").read_text(encoding="utf-8-sig"))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Subscript) and ast.unparse(node.func.value) == "JiT_models")
        for keyword in call.keywords:
            if keyword.arg in ("degradation_conditioning", "local_degradation_conditioning"):
                self.assertNotIn("psf_local", ast.literal_eval(keyword.value.comparators[0]))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Denoiser")
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        assignment = next(node for node in forward.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "net_degradation" for target in node.targets))
        self.assertEqual(ast.unparse(assignment.value.test), "self.degradation_mode == 'psf_local'")
        self.assertIsNone(assignment.value.body.value)
        generate = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
        branch = next(node for node in generate.body if isinstance(node, ast.If))
        self.assertNotIn("_degradation_probabilities", ast.unparse(ast.Module(body=branch.body, type_ignores=[])))

    def test_loss_gradient_path_and_identity_wiring(self):
        source = (ROOT / "degradation/a2_loss.py").read_text(encoding="utf-8-sig")
        self.assertIn("prediction_jit.float()", source)
        self.assertNotIn("prediction_jit.detach", source)
        self.assertNotIn(".clamp", source)
        self.assertNotIn("torch.no_grad()", source)
        self.assertIn("probabilities.detach().float()", source)
        self.assertIn("eligible = ~is_identity", source)
        engine = (ROOT / "engine_restoration.py").read_text(encoding="utf-8-sig")
        self.assertIn('forward_options["is_identity"] = batch["is_identity"]', engine)
        self.assertIn("**forward_options", engine)


if __name__ == "__main__":
    unittest.main()

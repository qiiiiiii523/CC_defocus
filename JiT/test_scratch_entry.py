"""Dependency-free entry-point checks. Does not import PyTorch or train a model."""
import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
TREE = ast.parse((ROOT / 'main_restoration.py').read_text(encoding='utf-8-sig'))
HELPERS = {'get_args_parser', '_resolve_init_mode', '_initialize_fresh_model',
           '_check_pair_counts', '_check_train_val_overlap', '_manifest_sha256'}


class ScratchEntryTests(unittest.TestCase):
    def setUp(self):
        self.loads = []
        def load(*args, **kwargs):
            self.loads.append(args)
            return {'model': 'official-state'}
        self.ns = dict(argparse=argparse, Path=Path, hashlib=hashlib,
                       JIT_ROOT=ROOT, PROJECT_ROOT=ROOT.parent,
                       torch=SimpleNamespace(load=load))
        nodes = [node for node in TREE.body
                 if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'entry-helpers', 'exec'), self.ns)

    def test_modes_and_parser_defaults(self):
        parser = self.ns['get_args_parser']()
        args = parser.parse_args([])
        self.assertEqual(args.expected_train_pairs, 2000)
        self.assertEqual(args.expected_val_pairs, 300)
        self.assertEqual(args.eval_weights, 'ema1')
        resolve = self.ns['_resolve_init_mode']
        self.assertEqual(resolve(None), 'pretrained')
        self.assertEqual(resolve('scratch'), 'scratch')
        self.assertEqual(resolve(None, {'args': SimpleNamespace(init_mode='scratch')}), 'scratch')
        self.assertEqual(resolve(None, {'args': {}}), 'pretrained')
        with self.assertRaises(ValueError):
            resolve('pretrained', {'args': {'init_mode': 'scratch'}})

    def test_scratch_never_reads_checkpoint(self):
        copied = []
        model = SimpleNamespace(net=SimpleNamespace(
            initialize_blur_condition_from_state_embedder=lambda: copied.append(True)))
        # Deliberately lacks is_file(): scratch must not even inspect this path.
        args = SimpleNamespace(init_mode='scratch', pretrained=object())
        self.ns['_initialize_fresh_model'](model, args)
        self.assertEqual(copied, [True])
        self.assertEqual(self.loads, [])

    def test_pretrained_still_loads(self):
        loaded = []
        model = SimpleNamespace(load_official_state_dict=loaded.append)
        args = SimpleNamespace(init_mode='pretrained',
                               pretrained=SimpleNamespace(is_file=lambda: True))
        self.ns['_initialize_fresh_model'](model, args)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(loaded, ['official-state'])

    def test_pair_counts(self):
        check = self.ns['_check_pair_counts']
        check(2000, 2000, 'train')
        check(65000, 0, 'train')
        for actual, expected in [(1, 2000), (0, 0), (3, -1)]:
            with self.assertRaises(ValueError):
                check(actual, expected, 'train')

    def test_overlap_ids_and_paths(self):
        def record(sample, blur, clear):
            return SimpleNamespace(sample_id=sample, blur_path=ROOT / blur, clear_path=ROOT / clear)
        train = [record('train', 'b1.png', 'c1.png')]
        check = self.ns['_check_train_val_overlap']
        check(train, [record('val', 'b2.png', 'c2.png')])
        for val in [record('train', 'b2.png', 'c2.png'), record('val', 'b1.png', 'c2.png')]:
            with self.assertRaises(ValueError):
                check(train, [val])

    def test_fingerprint_and_syntax(self):
        path = ROOT / 'main_restoration.py'
        self.assertEqual(self.ns['_manifest_sha256'](path),
                         hashlib.sha256(path.read_bytes()).hexdigest())
        for name in ('main_restoration.py', 'model_jit.py', 'engine_restoration.py'):
            compile((ROOT / name).read_text(encoding='utf-8-sig'), name, 'exec')


if __name__ == '__main__':
    unittest.main()

"""Offline policy and real-git forwarding regressions; no credentials or network needed."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import release_policy as policy


class ReleasePolicyTests(unittest.TestCase):
    def test_stable_versions_only(self):
        self.assertEqual(policy.version_line('0.145.2'), '0.145')
        for value in ['1.2', '1.2.3-beta', '01.2.3', 'latest', '../1.2.3']:
            with self.assertRaises(ValueError):
                policy.version_line(value)

    def test_metadata_excluded_but_mixed_files_fail(self):
        rules = {'metadata': ['version.txt', 'CHANGELOG.md'], 'mixed-version-files': ['package.json']}
        self.assertEqual(policy.split_paths(['fix.py', 'version.txt', 'CHANGELOG.md',
                         '.github/workflows/ci.yml', policy.LINE_PATH], rules), ['fix.py'])
        with self.assertRaisesRegex(ValueError, 'Manual forward-port'):
            policy.split_paths(['fix.py', 'package.json'], rules)

    def test_maintenance_cannot_cross_minor_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.github').mkdir()
            rules = {'trunk': 'main', 'adoption-version': '1.2.3', 'config': 'main.json',
                     'maintenance-config': 'patch.json', 'manifest': 'manifest.json'}
            (root / policy.POLICY_PATH).write_text(json.dumps(rules))
            (root / 'patch.json').write_text(json.dumps({'packages': {'.': {'versioning': 'always-bump-patch'}}}))
            (root / 'main.json').write_text(json.dumps({'packages': {'.': {'versioning': 'always-bump-minor'}}}))
            (root / policy.LINE_PATH).write_text('{"line":"1.2"}')
            (root / 'manifest.json').write_text('{".":"1.2.3"}')
            policy.validate(root, 'release/1.2')
            policy.validate(root, 'main')
            (root / 'manifest.json').write_text('{".":"1.2.4"}')
            with self.assertRaises(ValueError):
                policy.validate(root, 'main')
            (root / 'manifest.json').write_text('{".":"1.3.0"}')
            with self.assertRaises(ValueError):
                policy.validate(root, 'release/1.2')
            policy.validate(root, 'main')

    @contextlib.contextmanager
    def git_repo(self, conflict=False):
        with tempfile.TemporaryDirectory() as directory:
            old = Path.cwd()
            root = Path(directory)
            subprocess.run(['git', 'init', '--bare', str(root/'remote.git')], check=True, capture_output=True)
            subprocess.run(['git', 'init', '-b', 'main', str(root/'work')], check=True, capture_output=True)
            os.chdir(root/'work')
            try:
                policy.git('config', 'user.email', 'test@example.com')
                policy.git('config', 'user.name', 'Test')
                policy.git('remote', 'add', 'origin', str(root/'remote.git'))
                Path('code.txt').write_text('before\n')
                Path('version.txt').write_text('1.2.0\n')
                policy.git('add', '.')
                policy.git('commit', '-m', 'feat: baseline')
                policy.git('checkout', '-b', 'release/1.2')
                Path('code.txt').write_text('fixed\n')
                Path('version.txt').write_text('1.2.1\n')
                policy.git('add', '.')
                policy.git('commit', '-m', 'fix: repair bug')
                source = policy.git('rev-parse', 'HEAD').decode().strip()
                policy.git('push', 'origin', 'release/1.2')
                policy.git('checkout', 'main')
                Path('version.txt').write_text('1.3.0\n')
                if conflict:
                    Path('code.txt').write_text('different fix\n')
                policy.git('add', '.')
                policy.git('commit', '-m', 'feat: next release')
                trunk = policy.git('rev-parse', 'HEAD').decode().strip()
                policy.git('push', 'origin', 'main')
                yield source, trunk
            finally:
                os.chdir(old)

    def event(self, source):
        return {'pull_request': {'number': 42, 'merged': True, 'base': {'ref': 'release/1.2'},
                'head': {'ref': 'fix/bug'}, 'merge_commit_sha': source, 'labels': [],
                'html_url': 'https://github.com/example/repo/pull/42'}}

    def test_forward_port_has_trunk_parent_and_keeps_trunk_metadata(self):
        real_run = subprocess.run
        created = []
        def run(args, **kwargs):
            if args[:3] == ['gh', 'pr', 'create']:
                created.append(args)
                return subprocess.CompletedProcess(args, 0)
            return real_run(args, **kwargs)
        with self.git_repo() as (source, trunk), patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/repo'}), \
                patch.object(policy, 'gh', side_effect=[[], {'commits': 1}]), \
                patch.object(subprocess, 'run', side_effect=run):
            policy.forward_port(self.event(source), {'trunk': 'main', 'metadata': ['version.txt']})
            self.assertEqual(Path('code.txt').read_text(), 'fixed\n')
            self.assertEqual(Path('version.txt').read_text(), '1.3.0\n')
            self.assertEqual(policy.git('rev-parse', 'HEAD^').decode().strip(), trunk)
            self.assertEqual(len(created), 1)

    def test_conflict_never_pushes_partial_fix(self):
        with self.git_repo(conflict=True) as (source, _), patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/repo'}), \
                patch.object(policy, 'gh', side_effect=[[], {'commits': 1}]):
            with self.assertRaises(subprocess.CalledProcessError):
                policy.forward_port(self.event(source), {'trunk': 'main', 'metadata': ['version.txt']})
            self.assertEqual(policy.git('ls-remote', '--heads', 'origin', 'forwardport/pr-42'), b'')

    def test_multi_commit_rebase_refuses_partial_forward_port(self):
        with self.git_repo() as (first, _):
            policy.git('checkout', 'release/1.2')
            Path('second.txt').write_text('second fix\n')
            policy.git('add', '.')
            policy.git('commit', '-m', 'fix: second part')
            source = policy.git('rev-parse', 'HEAD').decode().strip()
            policy.git('push', 'origin', 'release/1.2')
            policy.git('checkout', 'main')
            responses = [[], {'commits': 2, 'head': {'sha': source}}, [{'sha': first}]]
            with patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/repo'}), \
                    patch.object(policy, 'gh', side_effect=responses):
                with self.assertRaisesRegex(ValueError, 'complete PR patch'):
                    policy.forward_port(self.event(source), {'trunk': 'main', 'metadata': ['version.txt']})
            self.assertEqual(policy.git('ls-remote', '--heads', 'origin', 'forwardport/pr-42'), b'')

    def test_squashed_multi_commit_fix_is_forwarded_completely(self):
        real_run = subprocess.run
        def run(args, **kwargs):
            if args[:3] == ['gh', 'pr', 'create']:
                return subprocess.CompletedProcess(args, 0)
            return real_run(args, **kwargs)
        with self.git_repo() as (first, _):
            policy.git('checkout', 'release/1.2')
            Path('second.txt').write_text('second fix\n')
            policy.git('add', '.')
            policy.git('commit', '-m', 'fix: second part')
            source = policy.git('rev-parse', 'HEAD').decode().strip()
            policy.git('push', 'origin', 'release/1.2')
            policy.git('checkout', '-b', 'squash', first + '^1')
            policy.git('merge', '--squash', 'release/1.2')
            policy.git('commit', '-m', 'fix: both parts')
            merged = policy.git('rev-parse', 'HEAD').decode().strip()
            policy.git('push', 'origin', 'squash')
            policy.git('checkout', 'main')
            responses = [[], {'commits': 2, 'head': {'sha': source}}, [{'sha': first}]]
            with patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/repo'}), \
                    patch.object(policy, 'gh', side_effect=responses), \
                    patch.object(subprocess, 'run', side_effect=run):
                policy.forward_port(self.event(merged), {'trunk': 'main', 'metadata': ['version.txt']})
            self.assertEqual(Path('code.txt').read_text(), 'fixed\n')
            self.assertEqual(Path('second.txt').read_text(), 'second fix\n')
            self.assertEqual(Path('version.txt').read_text(), '1.3.0\n')

    def test_release_pr_skipped(self):
        event = self.event('a'*40)
        event['pull_request']['head']['ref'] = 'release-please--branches--release/1.2'
        with patch.object(policy, 'git') as git:
            policy.forward_port(event, {'trunk': 'main'})
            git.assert_not_called()


if __name__ == '__main__':
    unittest.main()

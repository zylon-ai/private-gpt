const {test} = require('node:test');
const assert = require('node:assert/strict');
const check = require('./image-tag-policy.cjs');
const context = {repo: {owner: 'example', repo: 'component'}, sha: 'a'.repeat(40)};
function run(eventName, ref, payload = {}, status = 'behind') {
  const github = {rest: {repos: {compareCommitsWithBasehead: async args => {
    assert.equal(args.basehead, `main...${context.sha}`);
    if (status instanceof Error) throw status;
    return {data: {status}};
  }}}};
  return check({context: {...context, eventName, ref, payload}, github});
}
test('main branch builds may publish latest; maintenance, legacy, tags and PRs cannot', async () => {
  assert.equal(await run('workflow_dispatch', 'refs/heads/main'), true);
  assert.equal(await run('push', 'refs/heads/main'), true);
  for (const ref of ['refs/heads/release/1.2', 'refs/heads/zylon/1.58.0', 'refs/tags/v1.2.0']) {
    assert.equal(await run('workflow_dispatch', ref), false);
  }
  assert.equal(await run('pull_request', 'refs/heads/main'), false);
});
test('release must target main and its tagged commit must belong to main', async () => {
  for (const status of ['behind', 'identical']) {
    assert.equal(await run('release', 'refs/tags/v1.3.0', {release: {target_commitish: 'main'}}, status), true);
  }
  for (const status of ['ahead', 'diverged']) {
    assert.equal(await run('release', 'refs/tags/v1.2.1', {release: {target_commitish: 'main'}}, status), false);
  }
  assert.equal(await run('release', 'refs/tags/v1.2.1', {release: {target_commitish: 'release/1.2'}}), false);
  assert.equal(await run('release', 'refs/tags/v1.3.0-beta', {release: {target_commitish: 'main', prerelease: true}}), false);
  await assert.rejects(run('release', 'refs/tags/v1.3.0', {release: {target_commitish: 'main'}}, new Error('API unavailable')));
});
test('manual latest aliases and tag injection cannot bypass the policy', async () => {
  for (const name of ['tag', 'version']) {
    for (const tag of ['latest', 'latest-amd64', 'latest-arm64', 'latest.foo']) {
      await assert.rejects(run('workflow_dispatch', 'refs/heads/release/1.2', {inputs: {[name]: tag}}), /only.*main/);
      assert.equal(await run('workflow_dispatch', 'refs/heads/main', {inputs: {[name]: tag}}), true);
    }
    for (const tag of ['test1\nlatest', 'test1,type=raw,value=latest', ' latest ']) {
      await assert.rejects(run('workflow_dispatch', 'refs/heads/main', {inputs: {[name]: tag}}), /single Docker tag/);
    }
    assert.equal(await run('workflow_dispatch', 'refs/heads/release/1.2', {inputs: {[name]: '1.2.1'}}), false);
  }
});

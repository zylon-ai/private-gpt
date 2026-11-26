// Only main may publish latest (including architecture aliases). API failures stop publishing.
module.exports = async function imageTagPolicy({context, github}) {
  const event = context.payload;
  let allowed = ['push', 'workflow_dispatch'].includes(context.eventName) && context.ref === 'refs/heads/main';
  if (context.eventName === 'release') {
    const release = event.release || {};
    allowed = false;
    if (release.target_commitish === 'main' && !release.prerelease && !release.draft) {
      // target_commitish alone is insufficient for a manually created existing tag.
      const {data} = await github.rest.repos.compareCommitsWithBasehead({
        ...context.repo, basehead: `main...${context.sha}`,
      });
      allowed = ['behind', 'identical'].includes(data.status);
    }
  }
  // Validate before any image job, including raw tags that bypass metadata-action.
  if (context.eventName === 'workflow_dispatch') {
    for (const name of ['tag', 'version']) {
      const value = event.inputs?.[name];
      if (!value) continue;
      if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(value)) {
        throw new Error(`${name} must be a single Docker tag`);
      }
      if (/^latest(?:$|[-_.])/.test(value) && !allowed) {
        throw new Error('latest image tags may only be published from main');
      }
    }
  }
  return allowed;
};

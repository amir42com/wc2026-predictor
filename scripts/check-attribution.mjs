// Shared with amir42-web/scripts/check-attribution.mjs; keep the two in step.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

// Repository provenance (ATTR-001, META-001): commit messages and pull-request
// text carry no attribution of who or what produced the change. The patterns
// are structural, so this file names no tool: a co-author trailer, a
// session trailer, a "generated with/by" line, a hidden attribution marker,
// or a link to an agent session, project thread or agent task.
export const RULES = [
  {
    id: "co-author-trailer",
    pattern: /^\s*co-authored-by\s*:/i,
  },
  {
    id: "session-trailer",
    pattern: /^\s*[a-z][\w-]*-session\s*:\s*\S/i,
  },
  {
    id: "generated-by-line",
    pattern: /^[^\p{L}\p{N}]*generated (?:with|by)\b/iu,
  },
  {
    id: "attribution-marker",
    pattern: /<!--[^>]*attribution[^>]*-->/i,
  },
  {
    id: "session-link",
    pattern:
      /https?:\/\/\S*\/(?:code\/(?:session_|project\/chan_)|tasks\/task_)/i,
  },
];

// A head branch named after a tool reaches main in merge-commit subjects
// ("Merge pull request #N from owner/<branch>"), so the branch must carry a
// Conventional Commits type prefix (Dependabot's own branches excepted).
export const BRANCH_PATTERN =
  /^(?:feat|fix|chore|docs|refactor|test|ci|build|perf|style|revert|dependabot)\//;

export function checkBranch(branch) {
  if (!branch || BRANCH_PATTERN.test(branch)) return [];
  return [
    {
      source: "head branch",
      line: 1,
      rule: "branch-prefix",
      text: `${branch} (use feat/, fix/, chore/, docs/ or another Conventional Commits type)`,
    },
  ];
}

// Every commit carries the owner's own GitHub identity, name and address
// together. GitHub itself may commit (web merges and branch updates) and
// Dependabot may author its own updates; any other author or committer is a
// finding, because a local default identity shows on GitHub as somebody
// else's commit, and a tool's display name over the owner's address still
// shows in the history.
export const OWNER_NAME = "Amir";
export const OWNER_EMAIL = "155735614+amir42com@users.noreply.github.com";
export const ALLOWED_AUTHORS = [
  { name: OWNER_NAME, email: OWNER_EMAIL },
  {
    name: "dependabot[bot]",
    email: "49699333+dependabot[bot]@users.noreply.github.com",
  },
];
export const ALLOWED_COMMITTERS = [
  { name: OWNER_NAME, email: OWNER_EMAIL },
  { name: "GitHub", email: "noreply@github.com" },
];

const allowed = (list, name, email) =>
  list.some(
    (identity) =>
      identity.name === String(name ?? "").trim() &&
      identity.email ===
        String(email ?? "")
          .trim()
          .toLowerCase(),
  );

export function checkIdentity(commit) {
  const findings = [];
  const source = `commit ${commit.sha.slice(0, 7)}`;
  const owner = `${OWNER_NAME} <${OWNER_EMAIL}>`;

  if (!allowed(ALLOWED_AUTHORS, commit.authorName, commit.authorEmail)) {
    findings.push({
      source,
      line: 0,
      rule: "author-identity",
      text: `author ${commit.authorName ?? ""} <${commit.authorEmail}> (commit as ${owner})`,
    });
  }
  if (
    !allowed(ALLOWED_COMMITTERS, commit.committerName, commit.committerEmail)
  ) {
    findings.push({
      source,
      line: 0,
      rule: "committer-identity",
      text: `committer ${commit.committerName ?? ""} <${commit.committerEmail}> (commit as ${owner})`,
    });
  }

  return findings;
}

export function findAttribution(text, source) {
  const findings = [];
  const lines = String(text ?? "").split(/\r?\n/);

  lines.forEach((line, index) => {
    for (const rule of RULES) {
      if (rule.pattern.test(line)) {
        findings.push({ source, line: index + 1, rule: rule.id, text: line });
      }
    }
  });

  return findings;
}

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "..");
const ZERO_COMMIT = /^0+$/;

function commits(base, head) {
  const range =
    !base || ZERO_COMMIT.test(base) ? ["-1", head] : [`${base}..${head}`];
  const output = execFileSync(
    "git",
    ["log", "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e", ...range],
    { cwd: repositoryRoot, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
  );

  return output
    .split("\x1e")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [
        sha,
        authorName,
        authorEmail,
        committerName,
        committerEmail,
        message = "",
      ] = entry.split("\x1f");
      return {
        sha: sha.trim(),
        authorName,
        authorEmail,
        committerName,
        committerEmail,
        message,
      };
    });
}

function parseArguments(argv) {
  const result = { base: null, head: "HEAD" };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--base") result.base = argv[++index];
    else if (argument === "--head") result.head = argv[++index];
    else throw new Error(`Unknown argument: ${argument}`);
  }

  return result;
}

function main() {
  const { base, head } = parseArguments(process.argv.slice(2));
  const findings = [];

  for (const commit of commits(base, head)) {
    findings.push(...checkIdentity(commit));
    findings.push(
      ...findAttribution(commit.message, `commit ${commit.sha.slice(0, 7)}`),
    );
  }
  findings.push(...findAttribution(process.env.PR_TITLE, "pull-request title"));
  findings.push(...findAttribution(process.env.PR_BODY, "pull-request body"));
  findings.push(...checkBranch(process.env.PR_HEAD_REF));

  if (findings.length === 0) {
    console.log("Attribution check passed.");
    return;
  }

  console.error(
    "Attribution check failed. Commits and pull-request text name no tool, carry no attribution and use the owner's identity. Fix these:",
  );
  for (const finding of findings) {
    const where = finding.line > 0 ? `, line ${finding.line}` : "";
    console.error(
      `- ${finding.source}${where} [${finding.rule}]: ${finding.text.trim()}`,
    );
  }
  process.exitCode = 1;
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main();
}

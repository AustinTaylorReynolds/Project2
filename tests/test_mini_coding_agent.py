import contextlib
import io
import json
import pytest
from unittest.mock import patch

from mini_coding_agent import (
    FakeModelClient,
    MiniAgent,
    OllamaModelClient,
    SessionStore,
    Skill,
    SkillRegistry,
    WorkspaceContext,
    build_welcome,
    parse_frontmatter,
    run_skill_command,
)


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    kwargs.setdefault("skills", SkillRegistry())  # keep tests independent of any skills/ folder on disk
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".mini-coding-agent").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".mini-coding-agent" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_path_rejects_parent_escape(tmp_path):
    agent = build_agent(tmp_path, [])

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("outside-link/secret.txt")


def test_path_accepts_case_variant_on_case_insensitive_filesystems(tmp_path):
    project_root = tmp_path / "Proj"
    project_root.mkdir()
    agent = build_agent(project_root, [])
    variant = project_root.parent / project_root.name.lower() / "README.md"

    if not variant.exists():
        pytest.skip("case-sensitive filesystem")

    resolved = agent.path(str(variant))

    assert resolved.samefile(project_root / "README.md")


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "O   O" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_prompt_top_level_sections_stay_flush_left_with_multiline_content(tmp_path):
    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="fix/prompt-indentation",
        default_branch="main",
        status=" M mini_coding_agent.py\n?? tests/test_prompt.py",
        recent_commits=["abc123 first commit", "def456 second commit"],
        project_docs={"README.md": "line1\nline2"},
    )
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    agent = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.session["memory"] = {
        "task": "verify prompt formatting",
        "files": ["mini_coding_agent.py"],
        "notes": ["saw inconsistent indentation", "need regression coverage"],
    }
    agent.record({"role": "user", "content": "inspect prompt()", "created_at": "1"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "mini_coding_agent.py"},
            "content": "    def prompt(self, user_message):\n        ...",
            "created_at": "2",
        }
    )

    prompt = agent.prompt("is this issue legit?")
    lines = prompt.splitlines()

    for label in ["Rules:", "Tools:", "Valid response examples:", "Workspace:", "Memory:", "Transcript:", "Current user request:"]:
        assert label in lines
        assert f"            {label}" not in prompt


def _make_filler(i):
    return {"role": "tool", "name": "list_files", "args": {}, "content": "", "created_at": str(i)}


def test_history_text_deduplicates_reads_but_not_after_write(tmp_path):
    """read_file deduplication must not skip a read that follows a write.

    Realistic prior-turn history (non-recent window):
        user: "update config"
        assistant: <tool>read_file config</tool>
        tool:   config v1 (content: setting=true)
        assistant: <tool>write_file config</tool>
        tool:   wrote
        assistant: <tool>read_file config</tool>
        tool:   config v2 (content: setting=false)   <- MUST NOT be skipped

    Without fix: seen_reads={"config"} after first read; write does NOT clear it;
                 second read is wrongly skipped (LLM sees stale content).
    With fix: write clears seen_reads, second read is correctly shown.
    """
    agent = build_agent(tmp_path, [])

    # Simulate a prior turn with read->write->read on the same file
    # history_length=13, recent_start=7 (indices 0-6 non-recent, 7-12 recent)
    agent.record({"role": "user", "content": "update config", "created_at": "0"})        # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=true\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"write_file","args":{"path":"config.txt","content":"setting=false\n"}}</tool>', "created_at": "3"})
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "config.txt", "content": "setting=false\n"}, "content": "wrote config.txt", "created_at": "4"})  # index 4, non-recent
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "5"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=false\n", "created_at": "6"})  # index 6, non-recent, ADDED (write cleared dedup)
    # recent entries
    for i in range(7, 13):
        agent.record(_make_filler(i))

    history = agent.history_text()

    # Both read contents appear exactly once (check full line to avoid JSON false positives)
    assert "# config.txt\n   1: setting=true\n" in history
    assert "# config.txt\n   1: setting=false\n" in history
    # Also verify duplicate read (setting=true, same path) does NOT appear twice
    assert history.count("setting=true") == 1


def test_history_text_deduplicates_unchanged_repeated_reads(tmp_path):
    """read_file deduplication should still skip repeated reads with no write in between."""
    agent = build_agent(tmp_path, [])

    # Realistic: two identical reads with no write between them
    # history_length=10, recent_start=4 (indices 0-3 non-recent, 4-9 recent)
    agent.record({"role": "user", "content": "check logs", "created_at": "0"})  # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "log.txt"}, "content": "# log.txt\n   1: stable\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "3"})  # index 3, non-recent, SKIPPED (dup)
    for i in range(4, 10):
        agent.record(_make_filler(i))  # indices 4-9, recent

    history = agent.history_text()

    # Only first read should appear; duplicates must be skipped
    assert history.count("stable") == 1


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False
    assert captured["body"]["raw"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 42


# --------------------------------------------------------------------------
# 7) Agent skills
# --------------------------------------------------------------------------
ACTIVE_HEADER = "Active skills (follow these instructions for the current task):"
REVIEW_DESCRIPTION = "Review a pull request or local changes. Use when the user asks for a PR review or code review."


def make_skill(root, name="pr-review", description=REVIEW_DESCRIPTION, body="Step 1: run git diff.\nStep 2: report.", files=None, folder="skills", filename="SKILL.md"):
    skill_dir = root / folder / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    header = f"---\nname: {name}\ndescription: {description}\n---\n\n" if description else ""
    (skill_dir / filename).write_text(header + body + "\n", encoding="utf-8")
    for relative, text in (files or {}).items():
        target = skill_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return skill_dir


def build_skill_agent(tmp_path, outputs, **kwargs):
    return build_agent(tmp_path, outputs, skills=SkillRegistry.discover([tmp_path]), **kwargs)


def test_parse_frontmatter_handles_quotes_and_wrapped_values():
    meta, body = parse_frontmatter(
        "---\nname: \"pr-review\"\ndescription: >\n  first part\n  second part: with colon\n---\n# Title\nbody\n"
    )
    assert meta == {"name": "pr-review", "description": "first part second part: with colon"}
    assert body == "# Title\nbody"
    assert parse_frontmatter("# no header\ntext") == ({}, "# no header\ntext")
    assert parse_frontmatter("---\nunterminated: yes\n") == ({}, "---\nunterminated: yes")


def test_registry_discovers_skill_folders(tmp_path):
    make_skill(tmp_path, "pr-review")
    make_skill(tmp_path, "from-singular-dir", folder="skill", description="Found in the singular skill/ folder.")
    make_skill(tmp_path, "lower-case-file", filename="skill.md")
    make_skill(tmp_path, "No Frontmatter", description="", body="# Heading\nFirst prose line becomes the description.")
    (tmp_path / "skills" / "not-a-skill").mkdir()
    (tmp_path / "skills" / "not-a-skill" / "notes.txt").write_text("no SKILL.md here", encoding="utf-8")

    registry = SkillRegistry.discover([tmp_path])

    assert registry.names() == ["from-singular-dir", "lower-case-file", "no-frontmatter", "pr-review"]
    assert registry.find("PR_Review").name == "pr-review"
    assert registry.find("no-frontmatter").description == "First prose line becomes the description."
    assert registry.find("not-a-skill") is None


def test_registry_discovers_claude_skills_folder(tmp_path):
    make_skill(
        tmp_path,
        name="build-q-learning",
        folder=".claude/skills",
        description="Build the Lab 4 Q-learning vacuum robot.",
        body="Orchestrator steps.",
    )
    registry = SkillRegistry.discover([tmp_path])

    assert registry.find("build-q-learning") is not None


def test_registry_earlier_location_overrides_later_and_explicit_dir_wins(tmp_path):
    workspace, agent_home, custom = tmp_path / "ws", tmp_path / "home", tmp_path / "custom"
    make_skill(workspace, "demo", description="workspace version", body="workspace body")
    make_skill(agent_home, "demo", description="bundled version", body="bundled body")
    make_skill(agent_home, "only-bundled")
    make_skill(custom, "demo", description="custom version", body="custom body", folder=".")

    assert SkillRegistry.discover([workspace, agent_home]).find("demo").body == "workspace body"
    assert SkillRegistry.discover([workspace, agent_home]).find("only-bundled") is not None
    assert SkillRegistry.discover([workspace, agent_home], skills_dir=custom).find("demo").body == "custom body"


def test_agent_without_skills_has_no_skill_tool_or_prompt_sections(tmp_path):
    agent = build_agent(tmp_path, [])

    assert "use_skill" not in agent.tools
    assert "Skills:" not in agent.prefix
    assert ACTIVE_HEADER not in agent.prompt("hello").splitlines()
    assert "No skills found" in agent.skills_listing()


def test_catalog_and_use_skill_tool_are_in_the_stable_prefix(tmp_path):
    make_skill(tmp_path, body="SECRET-BODY-TEXT")
    agent = build_skill_agent(tmp_path, [])

    assert "use_skill" in agent.tools and agent.tools["use_skill"]["risky"] is False
    assert f"- pr-review: {REVIEW_DESCRIPTION}" in agent.prefix
    assert "Skills:" in agent.prefix.splitlines()
    assert "SECRET-BODY-TEXT" not in agent.prefix  # only name + description until activated


def test_model_can_activate_a_skill_during_execution(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(
        tmp_path,
        [
            '<tool>{"name":"use_skill","args":{"skill":"pr-review"}}</tool>',
            "<final>Reviewed.</final>",
        ],
    )

    answer = agent.ask("Take a look at the repo and tell me what you think")  # nothing in the request matches

    assert answer == "Reviewed."
    first_prompt, second_prompt = agent.model_client.prompts
    assert "FOLLOW-THESE-STEPS-MARKER" not in first_prompt
    assert ACTIVE_HEADER not in first_prompt.splitlines()
    assert ACTIVE_HEADER in second_prompt.splitlines() and "FOLLOW-THESE-STEPS-MARKER" in second_prompt
    assert agent.session["active_skills"] == ["pr-review"]


def test_matching_prompt_triggers_skill_before_the_first_model_call(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(tmp_path, ["<final>ok</final>"])
    events = []
    agent.skill_log = events.append

    agent.ask("Please review my pull request")

    assert "FOLLOW-THESE-STEPS-MARKER" in agent.model_client.prompts[0]
    assert agent.session["active_skills"] == ["pr-review"]
    assert events == ["activated 'pr-review' (matched the request)"]
    assert any(item["role"] == "tool" and item["name"] == "use_skill" for item in agent.session["history"])


def test_skill_name_in_request_triggers_it(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER", description="Something unrelated to the words below.")
    agent = build_skill_agent(tmp_path, ["<final>ok</final>"])

    agent.ask("run the PR review skill on the last commit")

    assert "FOLLOW-THESE-STEPS-MARKER" in agent.model_client.prompts[0]


def test_unrelated_prompt_does_not_trigger_skill(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(tmp_path, ["<final>ok</final>", "<final>ok</final>"])

    agent.ask("Fix the binary search bug in search.py")
    agent.ask("Review README.md")  # a single shared word is not enough

    assert all("FOLLOW-THESE-STEPS-MARKER" not in prompt for prompt in agent.model_client.prompts)
    assert agent.session["active_skills"] == []


def test_unknown_skill_returns_error_listing_available_skills(tmp_path):
    make_skill(tmp_path)
    agent = build_skill_agent(tmp_path, [])

    result = agent.run_tool("use_skill", {"skill": "does-not-exist"})
    empty = agent.run_tool("use_skill", {})

    assert result.startswith("error: invalid arguments for use_skill: unknown skill 'does-not-exist'")
    assert "available: pr-review" in result and "example:" in result
    assert "skill must not be empty" in empty
    assert agent.session["active_skills"] == []


def test_use_skill_reads_bundled_files_but_not_outside_the_skill_folder(tmp_path):
    make_skill(tmp_path, files={"checklist.md": "CHECKLIST-CONTENT", "scripts/run.py": "print(1)"})
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    agent = build_skill_agent(tmp_path, [])

    assert "CHECKLIST-CONTENT" in agent.run_tool("use_skill", {"skill": "pr-review", "file": "checklist.md"})
    assert "print(1)" in agent.run_tool("use_skill", {"skill": "pr-review", "file": "scripts/run.py"})
    assert "escapes skill folder" in agent.run_tool("use_skill", {"skill": "pr-review", "file": "../../secret.txt"})
    assert "no such file" in agent.run_tool("use_skill", {"skill": "pr-review", "file": "missing.md"})
    agent.activate_skill("pr-review")
    assert "Bundled files (read with use_skill file=...): checklist.md, scripts/run.py" in agent.active_skills_text()


def test_active_skill_instructions_are_not_clipped_by_history_compression(tmp_path):
    long_body = "\n".join(f"Step {i}: do something specific." for i in range(1, 120)) + "\nFINAL-STEP-MARKER"
    assert len(long_body) > 2500
    make_skill(tmp_path, body=long_body)
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    read = '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":%d}}</tool>'
    agent = build_skill_agent(tmp_path, [read % n for n in range(1, 6)] + ["<final>done</final>"], max_steps=6)

    agent.ask("Please review my pull request")

    assert len(agent.model_client.prompts) == 6
    assert all("FINAL-STEP-MARKER" in prompt for prompt in agent.model_client.prompts)


def test_active_skills_are_capped_persisted_and_cleared_on_reset(tmp_path):
    for name in ("alpha", "beta", "gamma", "delta"):
        make_skill(tmp_path, name, description=f"{name} skill")
    agent = build_skill_agent(tmp_path, [])

    for name in ("alpha", "beta", "gamma", "delta"):
        assert agent.activate_skill(name).name == name
    assert agent.session["active_skills"] == ["beta", "gamma", "delta"]  # oldest evicted
    agent.activate_skill("beta")
    assert agent.session["active_skills"] == ["gamma", "delta", "beta"]  # re-activation refreshes

    resumed = MiniAgent.from_session(
        FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"],
        approval_policy="auto", skills=agent.skills,
    )
    assert resumed.session["active_skills"] == ["gamma", "delta", "beta"]
    assert "[skill: beta]" in resumed.prompt("go")

    resumed.reset()
    assert resumed.session["active_skills"] == []
    assert ACTIVE_HEADER not in resumed.prompt("go").splitlines()


def test_sessions_saved_before_skills_existed_still_resume(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    agent.ask("Start a session")
    del agent.session["active_skills"]
    del agent.session["skill_logs"]
    agent.session_store.save(agent.session)

    resumed = MiniAgent.from_session(
        FakeModelClient([]), agent.workspace, agent.session_store, agent.session["id"],
        approval_policy="auto", skills=SkillRegistry(),
    )

    assert resumed.session["active_skills"] == []
    assert resumed.session["skill_logs"] == []


def test_delegate_child_inherits_skills_and_active_skills(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(tmp_path, ['<tool>{"name":"delegate","args":{"task":"look around","max_steps":2}}</tool>', "<final>child done</final>", "<final>parent done</final>"])
    agent.activate_skill("pr-review")

    agent.ask("Investigate")

    child_prompt = agent.model_client.prompts[1]
    assert "use_skill" in child_prompt and "FOLLOW-THESE-STEPS-MARKER" in child_prompt


def test_slash_commands_list_and_activate_skills(tmp_path):
    make_skill(tmp_path)
    agent = build_skill_agent(tmp_path, [])

    def run(text):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = run_skill_command(agent, text)
        return result, out.getvalue()

    assert run("/skills")[0] == (True, None)
    assert "pr-review" in run("/skills")[1]
    assert run("/skill-logs")[0] == (True, None)
    assert "No skill events logged yet" in run("/skill-logs")[1]
    assert run("/help")[0] == (False, None)
    assert run("/nothing")[0] == (False, None)

    assert run("/skill")[0] == (True, None)
    result, output = run("/skill nope")
    assert result == (True, None) and "unknown skill 'nope'" in output and "pr-review" in output

    result, output = run("/skill pr-review check my last commit")
    assert result == (True, "check my last commit") and "skill activated: pr-review" in output
    assert "slash_command" in run("/skill-logs")[1]
    agent.session["active_skills"].clear()
    assert run("/pr-review look at the diff")[0] == (True, "look at the diff")
    assert agent.session["active_skills"] == ["pr-review"]
    assert "* pr-review" in agent.skills_listing()


def test_skill_without_body_or_name_is_skipped(tmp_path):
    folder = tmp_path / "skills" / "empty"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: empty\ndescription: nothing inside\n---\n", encoding="utf-8")

    assert Skill.load(folder) is None
    assert len(SkillRegistry.discover([tmp_path])) == 0


def test_skill_files_with_windows_line_endings_are_parsed(tmp_path):
    skill_dir = tmp_path / "skills" / "win-skill"
    skill_dir.mkdir(parents=True)
    raw = "---\r\nname: win-skill\r\ndescription: Written on Windows with CRLF line endings\r\n---\r\n\r\n# Title\r\nStep one.\r\n"
    (skill_dir / "SKILL.md").write_bytes(raw.encode("utf-8"))

    skill = SkillRegistry.discover([tmp_path]).find("win-skill")

    assert skill.description == "Written on Windows with CRLF line endings"
    assert skill.body.splitlines() == ["# Title", "Step one."]


def test_skill_logging_records_prompt_similarity_and_executed_skill_on_match(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(tmp_path, ["<final>ok</final>"])

    agent.ask("Please review my pull request")

    assert len(agent.session["skill_logs"]) == 1
    entry = agent.session["skill_logs"][0]
    assert entry["trigger"] == "auto"
    assert entry["prompt"] == "Please review my pull request"
    assert entry["similarity_check"]["selected"] == "pr-review"
    assert entry["similarity_check"]["candidates"][0]["score"] >= 3
    assert entry["similarity_check"]["candidates"][0]["cosine_similarity"] > 0
    assert entry["executed_skill"]["name"] == "pr-review"

    lines = agent.skill_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["prompt"] == "Please review my pull request"
    assert logged["similarity_check"]["selected"] == "pr-review"
    assert logged["executed_skill"]["name"] == "pr-review"


def test_skill_logging_records_similarity_check_on_non_match_and_model_trigger(tmp_path):
    make_skill(tmp_path, body="FOLLOW-THESE-STEPS-MARKER")
    agent = build_skill_agent(
        tmp_path,
        [
            '<tool>{"name":"use_skill","args":{"skill":"pr-review"}}</tool>',
            "<final>Reviewed.</final>",
        ],
    )

    agent.ask("Take a look at the repo and tell me what you think")

    logs = agent.session["skill_logs"]
    assert [entry["trigger"] for entry in logs] == ["auto", "model_tool"]
    assert logs[0]["executed_skill"] is None
    assert logs[0]["similarity_check"]["selected"] is None
    assert logs[0]["similarity_check"]["candidates"][0]["matched"] is False
    assert logs[1]["executed_skill"]["name"] == "pr-review"
    assert logs[1]["prompt"] == "Take a look at the repo and tell me what you think"


def test_delegate_dispatches_named_claude_subagent_with_write_access(tmp_path):
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "q-table-setup.md").write_text(
        "---\nname: q-table-setup\n---\n\nWrite q_table.py with ACTIONS.\n",
        encoding="utf-8",
    )
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"agent":"q-table-setup","task":"Implement your component."}}</tool>',
            '<tool name="write_file" path="q_table.py"><content>ACTIONS = ["N", "S", "E", "W"]\n</content></tool>',
            "<final>q_table.py written.</final>",
            "<final>Delegation complete.</final>",
        ],
    )

    answer = agent.ask("Build q_table")

    assert answer == "Delegation complete."
    assert (tmp_path / "q_table.py").read_text(encoding="utf-8") == 'ACTIONS = ["N", "S", "E", "W"]\n'
    child_prompt = agent.model_client.prompts[1]
    assert "Sub-agent instructions (q-table-setup):" in child_prompt
    assert "Write q_table.py with ACTIONS." in child_prompt

# --------------------------------------------------------------------------
# Small-model robustness (found by reading real Lab 4 sessions)
# --------------------------------------------------------------------------
def make_named_subagent(tmp_path, name="q-table-setup", body="Write q_table.py with ACTIONS."):
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{name}.md").write_text(f"---\nname: {name}\n---\n\n{body}\n", encoding="utf-8")


def test_parse_accepts_attribute_only_tool_calls():
    task = "Implement your component."
    for raw in (
        f'<tool name="delegate" agent="q-table-setup" task="{task}"/>',
        f'<tool name="delegate" agent="q-table-setup" task="{task}" />',
        f'<tool name="delegate" agent="q-table-setup" task="{task}">',  # closing tag forgotten
    ):
        kind, payload = MiniAgent.parse(raw)
        assert kind == "tool", raw
        assert payload == {"name": "delegate", "args": {"agent": "q-table-setup", "task": task}}

    kind, payload = MiniAgent.parse('<tool name="delegate" agent="a" task="make x -> y"/>')
    assert kind == "tool" and payload["args"]["task"] == "make x -> y"  # '>' inside a quoted value

    kind, payload = MiniAgent.parse('<tool name="list_files" path="."/> <tool name="read_file" path="a.py"></tool>')
    assert kind == "tool" and payload["name"] == "list_files"  # the first call wins

    kind, payload = MiniAgent.parse('<tool name="delegate" agent="a" task="y"/>\nI will wait for the result.')
    assert kind == "tool" and payload["args"] == {"agent": "a", "task": "y"}  # trailing prose is not a body


def test_parse_never_acts_on_a_truncated_tool_body():
    for raw in (
        '<tool name="write_file" path="a.py"><content>def foo(',  # output cut off mid-file
        '<tool name="write_file" path="a.py">x = 1\ny = 2\n',  # body present, no closing tag
        '<tool agent="x" task="y"/>',  # no tool name
    ):
        kind, payload = MiniAgent.parse(raw)
        assert kind == "retry", raw
        assert "malformed tool output" in payload


def test_self_closing_delegate_call_runs_the_named_subagent(tmp_path):
    make_named_subagent(tmp_path)
    agent = build_agent(
        tmp_path,
        [
            '<tool name="delegate" agent="q-table-setup" task="Implement your component."/>',
            '<tool name="write_file" path="q_table.py"><content>ACTIONS = ["N", "S", "E", "W"]\n</content></tool>',
            "<final>q_table.py written.</final>",
            "<final>Delegation complete.</final>",
        ],
    )

    assert agent.ask("Build q_table") == "Delegation complete."
    assert (tmp_path / "q_table.py").exists()


def test_malformed_output_is_saved_for_debugging_but_never_shown_to_the_model(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="a.py"><content>SECRET_HALF_FILE = (',
            "<final>ok</final>",
        ],
    )

    assert agent.ask("Write a.py") == "ok"

    retry = next(item for item in agent.session["history"] if item["role"] == "assistant" and "raw" in item)
    assert "SECRET_HALF_FILE" in retry["raw"] and "malformed tool output" in retry["content"]
    assert "SECRET_HALF_FILE" not in agent.model_client.prompts[1]
    assert not (tmp_path / "a.py").exists()  # a half-written file was not saved


def test_named_subagent_keeps_a_minimum_step_budget_even_if_the_model_asks_for_less(tmp_path):
    make_named_subagent(tmp_path)
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            # the model copied max_steps=1 from the schema example; the sub-agent still needs several steps
            '<tool>{"name":"delegate","args":{"agent":"q-table-setup","task":"Do it.","max_steps":1}}</tool>',
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            '<tool name="write_file" path="q_table.py"><content>ACTIONS = ["N"]\n</content></tool>',
            "<final>q_table.py: all checks passed</final>",
            "<final>Delegation complete.</final>",
        ],
    )

    assert agent.ask("Build q_table") == "Delegation complete."
    assert (tmp_path / "q_table.py").exists()
    assert "Stopped after reaching the step limit" not in json.dumps(agent.session["history"])


def test_use_skill_when_already_active_tells_the_model_to_stop_repeating_it(tmp_path):
    make_skill(tmp_path)
    agent = build_skill_agent(tmp_path, [])

    first = agent.run_tool("use_skill", {"skill": "pr-review"})
    second = agent.run_tool("use_skill", {"skill": "pr-review"})

    assert "now active" in first
    assert "already active" in second and "Do not call use_skill for it again" in second

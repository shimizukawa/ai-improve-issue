"""Issue自動改善スクリプト - Pydantic AI版

Pydantic AI + pydantic-graph による実装

実行モード:
1. 通常モード: GitHub Actionsから自動実行（Issue作成時）
2. --dry-run: ローカル検証用（コメント投稿をスキップ）
3. --index-issues: RAGデータ生成モード（全Issueをベクトル化）
4. --update-single-issue N: 単一Issue更新モード
"""

import argparse
import asyncio

from pydantic_graph import GraphRunContext

from .config import AppSettings, EnvConfig, load_settings
from .tools import init_tools
from .workflow_graph import (
    create_issue_improvement_graph,
    create_rag_index_graph,
    create_single_issue_update_graph,
)
from .graph_nodes import (
    InputValidationNode,
    RAGIndexInitNode,
    SingleIssueUpdateNode,
    FetchIssueState,
    FetchAllIssuesState,
    GraphState,
)


async def run_normal_mode(
    args: argparse.Namespace,
    settings: AppSettings,
    env_config: EnvConfig,
    tools,
) -> None:
    """通常モード: Issue改善とコメント投稿"""
    if not env_config.issue_number:
        print("Error: ISSUE_NUMBER not set")
        return

    issue_number = int(env_config.issue_number)
    is_rag_enabled = env_config.is_rag_available and settings.rag.enabled

    state = GraphState(
        issue_number=issue_number,
        issue_title=env_config.issue_title,
        issue_body=env_config.issue_body,
        is_rag_enabled=is_rag_enabled,
        settings=settings,
        env_config=env_config,
        dry_run=args.dry_run,
    )
    graph = create_issue_improvement_graph()
    ctx = GraphRunContext(state)
    ctx.tools = tools
    result = await graph.run(InputValidationNode(), ctx)
    output = result.output
    status = output.get("status", "unknown")
    print(f"✓ Process completed: {status}")
    if output.get("chunks_indexed"):
        print(f"  Chunks indexed: {output['chunks_indexed']}")


async def index_all_issues(
    start: int, end: int | None, env_config: EnvConfig, tools
) -> None:
    """全Issueをインデックス登録（--index-issues モード）"""
    if not env_config.github_token or not env_config.github_repository:
        raise ValueError("GitHub token and repository are required")

    if not env_config.is_rag_available:
        raise ValueError("RAG environment variables are required")

    print("Fetching issues from GitHub...")

    state = FetchAllIssuesState(start=start, end=end, env_config=env_config)
    ctx = GraphRunContext(state)
    ctx.tools = tools
    graph = create_rag_index_graph()
    result = await graph.run(RAGIndexInitNode(), ctx)
    output = result.output
    if output.get("success", 0) == output.get("total", 0):
        print(f"✓ All {output['total']} issues indexed successfully")
    else:
        print(f"⚠ {output['success']}/{output['total']} issues indexed")


async def update_single_issue(issue_number: int, env_config: EnvConfig, tools) -> None:
    """単一Issueをインデックス更新（--update-single-issue モード）"""
    if not env_config.github_token or not env_config.github_repository:
        raise ValueError("GitHub token and repository are required")

    if not env_config.is_rag_available:
        raise ValueError("RAG environment variables are required")

    state = FetchIssueState(issue_number=issue_number, env_config=env_config)
    ctx = GraphRunContext(state)
    ctx.tools = tools
    graph = create_single_issue_update_graph()
    result = await graph.run(SingleIssueUpdateNode(), ctx)
    output = result.output
    print(
        f"✓ Issue #{output['issue_number']} updated ({output['chunks_indexed']} chunks indexed)"
    )


async def async_main():
    """メイン処理（async版）"""
    parser = argparse.ArgumentParser(
        description="Issue自動改善スクリプト (Pydantic AI版)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ローカル検証用（コメント投稿をスキップ）",
    )
    parser.add_argument(
        "--index-issues",
        action="store_true",
        help="RAGデータ生成モード（全Issueをベクトル化）",
    )
    parser.add_argument(
        "--update-single-issue",
        type=int,
        help="単一Issue更新モード（指定したIssue番号を更新）",
    )
    parser.add_argument(
        "--start", type=int, default=1, help="RAGインデックス開始Issue番号"
    )
    parser.add_argument("--end", type=int, help="RAGインデックス終了Issue番号")
    args = parser.parse_args()

    try:
        settings = load_settings()
        print("設定ファイルを読み込みました")
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return

    env_config = EnvConfig()
    tools = init_tools(env_config, settings)

    # RAGデータ生成モード
    if args.index_issues:
        try:
            await index_all_issues(args.start, args.end, env_config, tools)
        except ValueError as e:
            print(str(e))
        return

    # 単一Issue更新モード
    if args.update_single_issue:
        try:
            await update_single_issue(args.update_single_issue, env_config, tools)
        except ValueError as e:
            print(str(e))
        return

    # 通常モード
    await run_normal_mode(args, settings, env_config, tools)


def main():
    """同期版エントリポイント"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

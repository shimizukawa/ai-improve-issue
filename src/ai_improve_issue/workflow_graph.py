"""Workflow Graph - pydantic-graph実装"""

from pydantic_graph import Graph

from .config import AppSettings, EnvConfig
from .models import IssueData
from .graph_nodes import (
    GraphState,
    InputValidationNode,
    RAGSearchNode,
    TemplateDetectionNode,
    ContentGenerationNode,
    CommentFormattingNode,
    PostCommentNode,
    IndexCurrentIssueNode,
    RAGIndexState,
    RAGIndexInitNode,
    RAGIndexProcessNode,
    RAGIndexCompleteNode,
    SingleIssueUpdateState,
    SingleIssueUpdateNode,
    FetchAllIssuesNode,
    FetchIssueNode,
)


def create_issue_improvement_graph() -> Graph[GraphState, None, dict]:
    """Issue改善用のGraphを作成

    Args:
        settings: アプリケーション設定
        env_config: 環境変数設定

    Returns:
        Graph: Graphインスタンス（戻り値型: dict）
    """
    # Graphを作成
    graph = Graph(
        nodes=[
            InputValidationNode,
            RAGSearchNode,
            TemplateDetectionNode,
            ContentGenerationNode,
            CommentFormattingNode,
            PostCommentNode,
            IndexCurrentIssueNode,
        ],
        state_type=GraphState,
    )

    return graph


def create_rag_index_graph() -> Graph[RAGIndexState, None, dict]:
    """RAGインデックス用のGraphを作成

    Args:
    Returns:
        Graph: Graphインスタンス
    """
    return Graph(
        nodes=[
            FetchAllIssuesNode,
            RAGIndexInitNode,
            RAGIndexProcessNode,
            RAGIndexCompleteNode,
        ],
        state_type=RAGIndexState,
    )


def create_rag_index_state(
    issues: list[IssueData],
    settings: AppSettings,
    env_config: EnvConfig,
) -> RAGIndexState:
    """RAGインデックス用の状態を作成

    Args:
        issues: Issue一覧
        settings: アプリケーション設定
        env_config: 環境変数設定

    Returns:
        RAGIndexState: グラフの初期状態
    """
    return RAGIndexState(
        issues=issues,
        settings=settings,
        env_config=env_config,
    )


def create_single_issue_update_graph() -> Graph[SingleIssueUpdateState, None, dict]:
    """単一Issue更新用のGraphを作成

    Returns:
        Graph: Graphインスタンス
    """
    return Graph(
        nodes=[
            FetchIssueNode,
            SingleIssueUpdateNode,
        ],
        state_type=SingleIssueUpdateState,
    )


def create_single_issue_update_state(
    issue: IssueData,
    settings: AppSettings,
    env_config: EnvConfig,
) -> SingleIssueUpdateState:
    """単一Issue更新用の状態を作成

    Args:
        issue: Issue情報
        settings: アプリケーション設定
        env_config: 環境変数設定

    Returns:
        SingleIssueUpdateState: グラフの初期状態
    """
    return SingleIssueUpdateState(
        issue=issue,
        settings=settings,
        env_config=env_config,
    )

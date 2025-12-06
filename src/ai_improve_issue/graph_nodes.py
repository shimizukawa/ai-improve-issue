"""Graph処理ノード定義 - pydantic-graph実装"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated

from pydantic_graph import BaseNode, End, Edge, GraphRunContext

from .config import AppSettings, EnvConfig, load_template_content
from .llm_factory import LLMFactory
from .models import IssueData
from .rag_client import create_issue_chunks


@dataclass
class GraphState:
    """Graphの状態を管理"""

    issue_number: int
    issue_title: str
    issue_body: str
    is_rag_enabled: bool
    settings: AppSettings
    env_config: EnvConfig
    dry_run: bool = False

    # 中間結果
    similar_issues: list = field(default_factory=list)
    template_name: str | None = None
    improved_content: str | None = None


@dataclass
class FetchIssueState:
    """Issue取得の状態"""

    issue_number: int
    env_config: EnvConfig
    fetched_issue: IssueData | None = None


@dataclass
class FetchIssueNode(BaseNode[FetchIssueState, None, dict]):
    """Issue取得ノード"""

    async def run(
        self, ctx: GraphRunContext[FetchIssueState]
    ) -> Annotated[End[dict], Edge(label="Issue取得完了")]:
        """GitHub APIからIssue情報を取得"""
        print(f"Fetching issue #{ctx.state.issue_number}...")
        github_issue_tool = ctx.tools["github_issue"]
        issue = github_issue_tool.fetch_issue(ctx.state.issue_number)

        if not issue:
            return End({"status": "error", "issue_number": ctx.state.issue_number})

        ctx.state.fetched_issue = issue
        return End(
            {
                "status": "success",
                "issue_number": issue.number,
                "title": issue.title,
            }
        )


@dataclass
class FetchAllIssuesState:
    """全Issue取得の状態"""

    start: int
    end: int | None
    env_config: EnvConfig
    fetched_issues: list[IssueData] = field(default_factory=list)


@dataclass
class InputValidationNode(BaseNode[GraphState]):
    """入力バリデーションノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> (
        Annotated[RAGSearchNode, Edge(label="RAG検索実行")]
        | Annotated[TemplateDetectionNode, Edge(label="テンプレート判定へ")]
    ):
        """入力検証を実行"""
        print(f"Processing issue #{ctx.state.issue_number}")
        print(f"Title: {ctx.state.issue_title}")
        print(f"Body length: {len(ctx.state.issue_body)} characters")

        # 10文字未満の場合はスキップ
        combined = (ctx.state.issue_title or "") + (ctx.state.issue_body or "")
        text_without_spaces = (
            combined.replace(" ", "").replace("\n", "").replace("\t", "")
        )

        if len(text_without_spaces) < 10:
            raise ValueError(
                f"Issue #{ctx.state.issue_number} does not need improvement (too short)"
            )

        # RAG有効ならRAG検索へ、無効ならテンプレート判定へ
        if ctx.state.is_rag_enabled:
            return RAGSearchNode()
        else:
            return TemplateDetectionNode()


@dataclass
class RAGSearchNode(BaseNode[GraphState]):
    """RAG検索ノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> Annotated[TemplateDetectionNode, Edge(label="類似Issue検索完了")]:
        print("RAG mode: Enabled")

        embedding_tool = ctx.tools["embedding"]
        vector_search_tool = ctx.tools["vector_search"]
        SimilarIssue = ctx.state.settings.rag.similar_issue_class

        vector_search_tool.ensure_collection()
        query_text = f"{ctx.state.issue_title}\n{ctx.state.issue_body}"
        query_vector = embedding_tool.generate_embedding(query_text)
        similar_issues = vector_search_tool.search_similar_issues(
            query_vector,
            limit=ctx.state.settings.rag.top_k,
            exclude_issue_number=ctx.state.issue_number,
            SimilarIssue=SimilarIssue,
        )
        ctx.state.similar_issues = similar_issues
        if similar_issues:
            print(f"Found {len(similar_issues)} similar issues")
            for i, sim in enumerate(similar_issues, 1):
                print(
                    f"  {i}. #{sim.issue_number}: {sim.issue_title[:50]}... "
                    f"(similarity: {sim.similarity:.1%})"
                )
        else:
            print("No similar issues found")
        return TemplateDetectionNode()


@dataclass
class TemplateDetectionNode(BaseNode[GraphState]):
    """テンプレート判定ノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> Annotated[ContentGenerationNode, Edge(label="テンプレート判定完了")]:
        """テンプレート判定を実行"""
        # テンプレート情報を要約
        tmpl_summaries = []
        for name, tmpl in ctx.state.settings.templates.items():
            sp = (tmpl.system_prompt or "").strip()
            if len(sp) > 300:
                sp = sp[:300]
            kws = (tmpl.keywords or [])[:10]
            tmpl_summaries.append(
                {
                    "name": name,
                    "keywords": kws,
                    "system_prompt": sp,
                }
            )

        prompt = (
            "【Issue】\n"
            f"タイトル: {ctx.state.issue_title}\n"
            f"本文: {ctx.state.issue_body}\n\n"
            "【テンプレート候補一覧(JSON)】\n"
            f"{json.dumps(tmpl_summaries, ensure_ascii=False)}\n\n"
            "以下の形式で厳密に1件のみ出力してください。\n"
            '{"template": "<name>"}'
        )

        # Pydantic AI Agent実行
        print("LLM: Detecting template...")
        agent = LLMFactory.create_template_detection_agent(ctx.state.settings)
        result = await agent.run(prompt)

        template_name = result.data.template
        if not ctx.state.settings.validate_template(template_name):
            print(
                f"不明なテンプレート名 '{template_name}'。デフォルトテンプレートを使用します。"
            )
            template_name = ctx.state.settings.default_template

        ctx.state.template_name = template_name
        print(f"Detected template: {template_name}")

        return ContentGenerationNode()


@dataclass
class ContentGenerationNode(BaseNode[GraphState]):
    """文章生成ノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> Annotated[CommentFormattingNode, Edge(label="LLM文章生成完了")]:
        """文章生成を実行"""
        if not ctx.state.template_name:
            ctx.state.template_name = ctx.state.settings.default_template

        tmpl = ctx.state.settings.templates[ctx.state.template_name]
        template_content = load_template_content(tmpl)

        system_prompt = f"""
{tmpl.system_prompt}

【Issue記述】について、【類似する過去Issue】の内容を参考にして、概要を敬体でまとめてください。
また、【類似する過去Issue】に類似する内容があるか判定し、【## 類似Issue】セクションにそのidを出力してください。
もし存在しない場合は「なし」と出力してください。
出力形式以外の文章は不要です。
"""

        prompt = f"""
【Issue記述】
タイトル: {ctx.state.issue_title}
本文: {ctx.state.issue_body}

【テンプレート】
{template_content}
"""

        # RAG検索結果があれば追加
        if ctx.state.similar_issues and len(ctx.state.similar_issues) > 0:
            similar_info = (
                "\n\n【類似する過去Issue】\n以下の過去Issueを参考にしてください：\n"
            )
            for i, issue in enumerate(ctx.state.similar_issues, 1):
                similar_info += f"""
【参考Issue {i}】
- タイトル: {issue.issue_title}
- 本文抜粋: {issue.issue_body[:200]}...
- 類似度: {issue.similarity:.1%}
"""
            similar_info += "\n上記の参考Issueから、記述スタイルや必要な情報項目を学び、より具体的で実用的な例文を生成してください。"
            prompt += similar_info

        # Pydantic AI Agent実行
        print("LLM: Generating content...")
        agent = LLMFactory.create_content_generation_agent(
            ctx.state.settings, system_prompt
        )
        result = await agent.run(prompt)

        ctx.state.improved_content = result.data
        print("Content generated successfully")

        return CommentFormattingNode()


@dataclass
class CommentFormattingNode(BaseNode[GraphState]):
    """コメントフォーマットノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> Annotated[PostCommentNode, Edge(label="コメントフォーマット完了")]:
        """コメントフォーマットを実行"""
        print("Formatting comment...")

        template_display_names = {
            "feature_request": "機能要件",
            "bug_report": "バグ報告",
        }
        template_name = ctx.state.template_name or ctx.state.settings.default_template
        template_display = template_display_names.get(template_name, template_name)

        comment = f"""## 🤖 AIによるIssue記入例

**選定テンプレート**: {template_display}

---

{ctx.state.improved_content}

---
"""

        # RAG検索結果があれば追加
        if ctx.state.similar_issues and len(ctx.state.similar_issues) > 0:
            comment += "\n### 📚 参考にした類似Issue\n\nこの例文は以下の過去Issueを参考に生成しています：\n\n"
            for i, issue in enumerate(ctx.state.similar_issues, 1):
                comment += f"""{i}. **#{issue.issue_number}: {issue.issue_title}** ({issue.state})
   - 類似度: {issue.similarity:.0%}
   - {issue.url}

"""
            comment += "---\n\n"

        comment += (
            """💡 **使い方**: 上記の例文を参考に、Issue本文を編集してください。"""
        )
        if ctx.state.similar_issues and len(ctx.state.similar_issues) > 0:
            comment += "類似Issueも確認すると、より詳細な情報が得られます。"
        else:
            comment += "実際のプロジェクトに合わせて内容を修正してください。"

        comment += "\n\n<!-- AI-generated comment -->\n"

        # 状態にコメントを保存
        ctx.state.improved_content = comment

        return PostCommentNode()


@dataclass
class PostCommentNode(BaseNode[GraphState, None, dict]):
    """コメント投稿ノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> (
        Annotated[IndexCurrentIssueNode, Edge(label="RAG登録処理へ")]
        | Annotated[End[dict], Edge(label="処理完了")]
    ):
        """コメント投稿と条件分岐"""
        comment = ctx.state.improved_content

        # dry-runモード: コメント投稿をスキップ
        if ctx.state.dry_run:
            print("\n" + "=" * 60)
            print("[DRY RUN] コメント投稿をスキップします")
            print("=" * 60)
            print(comment)
            print("=" * 60)
            return End(
                {
                    "status": "dry_run",
                    "issue_number": ctx.state.issue_number,
                    "comment_length": len(comment),
                }
            )

        # 通常モード: GitHub CLI経由でコメント投稿
        print(f"Posting comment to issue #{ctx.state.issue_number}...")

        github_issue_tool = ctx.tools["github_issue"]
        github_issue_tool.post_comment(ctx.state.issue_number, comment)

        # RAG有効判定
        if ctx.state.is_rag_enabled:
            return IndexCurrentIssueNode()
        else:
            return End(
                {
                    "status": "comment_posted",
                    "issue_number": ctx.state.issue_number,
                    "comment_length": len(comment),
                }
            )


@dataclass
class IndexCurrentIssueNode(BaseNode[GraphState, None, dict]):
    """通常モード後のRAG登録ノード"""

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> Annotated[End[dict], Edge(label="インデックス登録完了")]:
        """現在のIssueをRAGにインデックス"""
        print("Indexing current issue to RAG...")

        issue = IssueData(
            number=ctx.state.issue_number,
            title=ctx.state.issue_title,
            body=ctx.state.issue_body,
            state="open",
            url=(
                f"https://github.com/{ctx.state.env_config.github_repository}/issues/{ctx.state.issue_number}"
                if ctx.state.env_config.github_repository
                else ""
            ),
            labels=[],
        )

        embedding_tool = ctx.tools["embedding"]
        vector_search_tool = ctx.tools["vector_search"]
        vector_search_tool.ensure_collection()
        chunks = create_issue_chunks(issue.title, issue.body)
        vectors = embedding_tool.generate_embeddings_batch(chunks)
        vector_search_tool.upsert_issue_chunks(
            issue_number=issue.number,
            chunks=chunks,
            vectors=vectors,
            title=issue.title,
            state=issue.state,
            url=issue.url,
            labels=issue.labels,
        )
        print(f"✓ Issue indexed successfully ({len(chunks)} chunks)")

        return End(
            {
                "status": "indexed",
                "issue_number": ctx.state.issue_number,
                "chunks_indexed": len(chunks),
            }
        )


# ===== RAG Indexing Mode =====


@dataclass
class RAGIndexState:
    """RAGインデックスモードの状態"""

    issues: list[IssueData]
    settings: AppSettings
    env_config: EnvConfig
    current_index: int = 0

    # 中間結果
    success_count: int = 0
    total_issues: int = field(default_factory=lambda: 0)

    def __post_init__(self):
        if self.total_issues == 0:
            self.total_issues = len(self.issues)


@dataclass
class RAGIndexInitNode(BaseNode[RAGIndexState]):
    """RAGインデックス初期化ノード"""

    async def run(
        self, ctx: GraphRunContext[RAGIndexState]
    ) -> Annotated[RAGIndexProcessNode, Edge(label="インデックス処理開始")]:
        """Qdrant接続確認"""
        print("=== RAG Indexing Mode ===")
        print(f"Total issues to index: {ctx.state.total_issues}")

        vector_search_tool = ctx.tools["vector_search"]
        vector_search_tool.ensure_collection()

        return RAGIndexProcessNode()


@dataclass
class RAGIndexProcessNode(BaseNode[RAGIndexState]):
    """RAGインデックス処理ノード"""

    async def run(
        self, ctx: GraphRunContext[RAGIndexState]
    ) -> (
        Annotated[RAGIndexProcessNode, Edge(label="次のIssue処理")]
        | Annotated[RAGIndexCompleteNode, Edge(label="インデックス完了")]
    ):
        """Issue単位のインデックス処理"""
        if ctx.state.current_index >= ctx.state.total_issues:
            return RAGIndexCompleteNode()

        issue = ctx.state.issues[ctx.state.current_index]
        print(
            f"[{ctx.state.current_index + 1}/{ctx.state.total_issues}] "
            f"Indexing issue #{issue.number}..."
        )

        embedding_tool = ctx.tools["embedding"]
        vector_search_tool = ctx.tools["vector_search"]
        chunks = create_issue_chunks(issue.title, issue.body)
        vectors = embedding_tool.generate_embeddings_batch(chunks)
        print(f"Qdrant: Upserting issue #{issue.number}...")
        vector_search_tool.upsert_issue_chunks(
            issue_number=issue.number,
            chunks=chunks,
            vectors=vectors,
            title=issue.title,
            state=issue.state,
            url=issue.url,
            labels=issue.labels,
        )

        # 次のIssueへ
        ctx.state.current_index += 1
        ctx.state.success_count += 1

        return RAGIndexProcessNode()


@dataclass
class RAGIndexCompleteNode(BaseNode[RAGIndexState, None, dict]):
    """RAGインデックス完了ノード"""

    async def run(
        self, ctx: GraphRunContext[RAGIndexState]
    ) -> Annotated[End[dict], Edge(label="インデックス完了")]:
        """完了ログ出力"""
        print("\n=== Indexing Complete ===")
        print(f"Success: {ctx.state.success_count}/{ctx.state.total_issues} issues")

        return End(
            {
                "success": ctx.state.success_count,
                "total": ctx.state.total_issues,
            }
        )


# ===== Single Issue Update Mode =====


@dataclass
class SingleIssueUpdateState:
    """単一Issue更新モードの状態"""

    issue: IssueData
    settings: AppSettings
    env_config: EnvConfig


@dataclass
class SingleIssueUpdateNode(BaseNode[SingleIssueUpdateState, None, dict]):
    """単一Issue更新ノード"""

    async def run(
        self, ctx: GraphRunContext[SingleIssueUpdateState]
    ) -> Annotated[End[dict], Edge(label="Issue更新完了")]:
        """単一IssueをインデックスBroadcast"""
        issue = ctx.state.issue
        print(f"=== Update Single Issue #{issue.number} ===")

        embedding_tool = ctx.tools["embedding"]
        vector_search_tool = ctx.tools["vector_search"]
        vector_search_tool.ensure_collection()
        chunks = create_issue_chunks(issue.title, issue.body)
        vectors = embedding_tool.generate_embeddings_batch(chunks)
        print(f"Qdrant: Upserting issue #{issue.number}...")
        vector_search_tool.upsert_issue_chunks(
            issue_number=issue.number,
            chunks=chunks,
            vectors=vectors,
            title=issue.title,
            state=issue.state,
            url=issue.url,
            labels=issue.labels,
        )
        print(f"Issue #{issue.number} updated successfully")

        return End({"issue_number": issue.number, "chunks_indexed": len(chunks)})


@dataclass
class FetchAllIssuesNode(BaseNode[FetchAllIssuesState, None, dict]):
    """全Issue取得ノード"""

    async def run(
        self, ctx: GraphRunContext[FetchAllIssuesState]
    ) -> Annotated[End[dict], Edge(label="全Issue取得完了")]:
        """GitHub APIから全Issue情報を取得"""
        print(f"Fetching issues from {ctx.state.start} to {ctx.state.end}...")
        github_issue_tool = ctx.tools["github_issue"]
        issues = github_issue_tool.fetch_all_issues(ctx.state.start, ctx.state.end)

        ctx.state.fetched_issues = issues
        return End({"status": "success", "count": len(issues)})

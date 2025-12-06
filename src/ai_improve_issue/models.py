"""Pydantic models定義"""

from datetime import datetime

from pydantic import BaseModel, Field


class TemplateSelection(BaseModel):
    """テンプレート判定結果"""

    template: str = Field(description="選択されたテンプレート名")


class SimilarIssue(BaseModel):
    """類似Issue情報"""

    issue_number: int
    issue_title: str
    issue_body: str
    state: str
    url: str
    similarity: float


class ProcessContext(BaseModel):
    """処理全体のコンテキスト"""

    # 入力情報
    issue_number: int
    issue_title: str
    issue_body: str

    # 実行モード
    mode: str = Field(
        default="normal"
    )  # normal, dry_run, index_issues, update_single_issue
    dry_run: bool = False

    # RAG設定
    is_rag_enabled: bool = False

    # 中間結果（各ノードで更新）
    template_name: str | None = None
    similar_issues: list[SimilarIssue] = Field(default_factory=list)
    improved_content: str | None = None
    formatted_comment: str | None = None

    # メタデータ
    execution_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class IssueChunk(BaseModel):
    """Issueチャンク情報"""

    issue_number: int
    chunk_index: int
    issue_title: str
    issue_body_chunk: str
    state: str
    url: str
    labels: list[str] = Field(default_factory=list)


class IssueData(BaseModel):
    """GitHub Issue情報"""

    number: int
    title: str
    body: str
    state: str
    url: str
    labels: list[str] = Field(default_factory=list)

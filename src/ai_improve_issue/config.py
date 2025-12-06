"""設定管理"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM設定"""

    provider: str = "google"
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 5000


class ProviderConfig(BaseModel):
    """プロバイダー設定"""

    api_key_env: str
    models: list[str]


class RAGConfig(BaseModel):
    """RAG設定"""

    enabled: bool = True
    embedding_dimensions: int = 256
    similarity_threshold: float = 0.5
    top_k: int = 3


class TemplateConfig(BaseModel):
    """テンプレート設定"""

    issue_template_file: str
    system_prompt: str
    keywords: list[str]


class AppSettings(BaseModel):
    """アプリケーション設定"""

    llm: LLMConfig
    providers: dict[str, ProviderConfig]
    rag: RAGConfig
    default_template: str
    templates: dict[str, TemplateConfig]

    def get_api_key(self) -> str:
        """現在のプロバイダーのAPIキーを取得"""
        provider_config = self.providers.get(self.llm.provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {self.llm.provider}")

        api_key = os.environ.get(provider_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {provider_config.api_key_env}"
            )

        return api_key

    def validate_template(self, template_name: str) -> bool:
        """テンプレート名が有効かチェック"""
        return template_name in self.templates


class EnvConfig(BaseModel):
    """環境変数設定"""

    github_repository: str = Field(
        default_factory=lambda: os.environ.get("GITHUB_REPOSITORY", "")
    )
    github_token: str = Field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN", "")
    )
    issue_number: str = Field(
        default_factory=lambda: os.environ.get("ISSUE_NUMBER", "")
    )
    issue_title: str = Field(default_factory=lambda: os.environ.get("ISSUE_TITLE", ""))
    issue_body: str = Field(default_factory=lambda: os.environ.get("ISSUE_BODY", ""))

    qdrant_url: str = Field(default_factory=lambda: os.environ.get("QDRANT_URL", ""))
    qdrant_api_key: str = Field(
        default_factory=lambda: os.environ.get("QDRANT_API_KEY", "")
    )
    voyage_api_key: str = Field(
        default_factory=lambda: os.environ.get("VOYAGE_API_KEY", "")
    )
    qdrant_collection_name: str = Field(
        default_factory=lambda: os.environ.get(
            "QDRANT_COLLECTION_NAME", "ai-improve-issues"
        )
    )

    @property
    def is_rag_available(self) -> bool:
        """RAG機能が利用可能かチェック"""
        return bool(self.qdrant_url and self.qdrant_api_key and self.voyage_api_key)


def find_repo_root() -> Path:
    """リポジトリルートを探索"""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return current


def load_settings() -> AppSettings:
    """設定ファイルを読み込む"""
    config_path = os.environ.get("AI_IMPROVE_ISSUE_CONFIG")
    if config_path:
        config_file = Path(config_path)
    else:
        repo_root = find_repo_root()
        config_file = repo_root / ".ai_improve_issue.yml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_file}\n"
            f"環境変数 AI_IMPROVE_ISSUE_CONFIG で設定ファイルパスを指定するか、\n"
            f"リポジトリルートに .ai_improve_issue.yml を配置してください。"
        )

    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppSettings(**data)


def load_template_content(template_config: TemplateConfig) -> str:
    """ISSUE_TEMPLATEファイルから実際のテンプレート内容を読み込む"""
    repo_root = find_repo_root()
    template_file = (
        repo_root
        / ".github"
        / "ISSUE_TEMPLATE"
        / f"{template_config.issue_template_file}.md"
    )

    if not template_file.exists():
        raise FileNotFoundError(
            f"Issueテンプレートファイルが見つかりません: {template_file}"
        )

    with open(template_file, encoding="utf-8") as f:
        content = f.read()

    # frontmatter除去
    lines = content.split("\n")
    if lines and lines[0] == "---":
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i] == "---":
                end_idx = i
                break
        if end_idx is not None:
            content = "\n".join(lines[end_idx + 1 :])

    return content.strip()

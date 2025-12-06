"""LLM Factory - 複数プロバイダー対応"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from .config import AppSettings
from .models import TemplateSelection


class LLMFactory:
    """LLM Agentファクトリー"""

    @staticmethod
    def create_model(settings: AppSettings) -> Model:
        """設定に応じたModelインスタンスを作成

        Args:
            settings: アプリケーション設定

        Returns:
            Pydantic AI Model インスタンス
        """
        llm_config = settings.llm
        # APIキーの存在確認
        _ = settings.get_api_key()

        # Pydantic AIは文字列でモデル名を指定すると自動でプロバイダーを判定
        # 例: "gemini-2.0-flash-exp" → Google, "gpt-4o" → OpenAI
        return llm_config.model

    @staticmethod
    def create_template_detection_agent(
        settings: AppSettings,
    ) -> Agent[None, TemplateSelection]:
        """テンプレート判定Agent作成

        Args:
            settings: アプリケーション設定

        Returns:
            テンプレート判定Agent
        """
        model = LLMFactory.create_model(settings)

        system_prompt = (
            "あなたはIssueの内容から最適なテンプレートを1つだけ選ぶ分類器です。\n"
            "提供されたテンプレート候補の中から、記述の目的・構造に最も適合するものを厳密に1件選んでください。"
        )

        return Agent(
            model=model,
            result_type=TemplateSelection,
            system_prompt=system_prompt,
            model_settings={
                "temperature": 0.1,
                "max_tokens": 256,
            },
        )

    @staticmethod
    def create_content_generation_agent(
        settings: AppSettings, system_prompt: str
    ) -> Agent[None, str]:
        """文章生成Agent作成

        Args:
            settings: アプリケーション設定
            system_prompt: システムプロンプト

        Returns:
            文章生成Agent
        """
        model = LLMFactory.create_model(settings)

        return Agent(
            model=model,
            result_type=str,
            system_prompt=system_prompt,
            model_settings={
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens,
            },
        )

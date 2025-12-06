"""AI関連ツールの初期化・ラッパー定義"""

import json
import subprocess
import tempfile
from typing import Any

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from .models import IssueData


class VoyageEmbeddingTool:
    def __init__(self, api_key: str, embedding_dimensions: int):
        self.client = voyageai.Client(api_key=api_key)
        self.embedding_dimensions = embedding_dimensions

    def generate_embedding(self, text: str) -> list[float]:
        result = self.client.embed(
            texts=[text],
            model="voyage-3.5-lite",
            output_dimension=self.embedding_dimensions,
        )
        return result.embeddings[0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(
            texts=texts,
            model="voyage-3.5-lite",
            output_dimension=self.embedding_dimensions,
        )
        return result.embeddings


class QdrantVectorSearchTool:
    def __init__(
        self, url: str, api_key: str, collection_name: str, embedding_dimensions: int
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
        self.Distance = Distance
        self.PayloadSchemaType = PayloadSchemaType
        self.VectorParams = VectorParams

    def ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.embedding_dimensions, distance=self.Distance.COSINE
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="issue_number",
                field_schema=self.PayloadSchemaType.INTEGER,
            )

    def search_similar_issues(
        self, query_vector, limit, exclude_issue_number, SimilarIssue
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit * 5,
        )
        points = getattr(response, "points", [])
        if not points:
            return []
        issue_map = {}
        for result in points:
            issue_num = result.payload.get("issue_number")
            if exclude_issue_number is not None and issue_num == exclude_issue_number:
                continue
            if (
                issue_num not in issue_map
                or result.score > issue_map[issue_num]["similarity"]
            ):
                issue_body = result.payload.get(
                    "issue_body_chunk"
                ) or result.payload.get("issue_body", "")
                issue_map[issue_num] = SimilarIssue(
                    issue_number=issue_num,
                    issue_title=result.payload.get("issue_title", ""),
                    issue_body=issue_body[:500],
                    state=result.payload.get("state", ""),
                    url=result.payload.get("url", ""),
                    similarity=result.score,
                )
        similar_issues = sorted(
            issue_map.values(), key=lambda x: x.similarity, reverse=True
        )[:limit]
        return similar_issues

    def upsert_issue_chunks(
        self, issue_number, chunks, vectors, title, state, url, labels
    ):
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            PointIdsList,
            PointStruct,
        )
        import uuid

        ids_to_delete = []
        offset = None
        while True:
            existing_points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="issue_number",
                            match=MatchValue(value=issue_number),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not existing_points:
                break
            ids_to_delete.extend(str(point.id) for point in existing_points)
            if next_offset is None:
                break
            offset = next_offset
        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "issue_number": issue_number,
                    "chunk_index": i,
                    "issue_title": title,
                    "issue_body_chunk": chunk,
                    "state": state,
                    "url": url,
                    "labels": labels,
                },
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)


class GitHubIssueTool:
    def __init__(self, github_token: str, github_repository: str):
        self.github_token = github_token
        self.github_repository = github_repository

    def fetch_issue(self, issue_number: int) -> "IssueData | None":
        """GitHub APIからIssue情報を取得"""
        if not self.github_repository:
            print("Error: GITHUB_REPOSITORY not set")
            return None
        cmd = [
            "gh",
            "api",
            f"/repos/{self.github_repository}/issues/{issue_number}",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env={
                "GH_TOKEN": self.github_token,
                "GH_REPO": self.github_repository,
            },
        )
        issue_data = json.loads(result.stdout)
        labels = [label["name"] for label in issue_data["labels"]]
        from .models import IssueData

        return IssueData(
            number=int(issue_data["number"]),
            title=issue_data["title"],
            body=issue_data["body"] or "",
            state=issue_data["state"],
            url=issue_data["html_url"],
            labels=labels,
        )

    def fetch_all_issues(self, start: int, end: int | None) -> list:
        """全Issue情報を取得"""
        if not self.github_repository:
            print("Error: GITHUB_REPOSITORY not set")
            return []
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env={
                "GH_TOKEN": self.github_token,
                "GH_REPO": self.github_repository,
            },
        )
        issues_data = json.loads(result.stdout)
        issue_numbers = [issue["number"] for issue in issues_data]
        if end is not None:
            issue_numbers = [n for n in issue_numbers if start <= n <= end]
        else:
            issue_numbers = [n for n in issue_numbers if n >= start]
        issues = []
        for num in issue_numbers:
            issue = self.fetch_issue(num)
            if issue:
                issues.append(issue)
        return issues

    def post_comment(self, issue_number: int, content: str) -> None:
        """GitHub CLI経由でコメントを投稿"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".md", delete=False
        ) as f:
            f.write(content)
            f.flush()
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(issue_number),
                    "--body-file",
                    f.name,
                ],
                check=True,
                env={
                    "GH_TOKEN": self.github_token,
                    "GH_REPO": self.github_repository,
                },
            )
        print(f"Comment posted successfully to issue #{issue_number}")


def init_tools(env_config, settings) -> dict:
    """
    必要なAIツールを初期化してdictで返す
    """
    tools: dict[str, Any] = {}
    tools["embedding"] = VoyageEmbeddingTool(
        api_key=env_config.voyage_api_key,
        embedding_dimensions=settings.rag.embedding_dimensions,
    )
    tools["vector_search"] = QdrantVectorSearchTool(
        url=env_config.qdrant_url,
        api_key=env_config.qdrant_api_key,
        collection_name=env_config.qdrant_collection_name,
        embedding_dimensions=settings.rag.embedding_dimensions,
    )
    tools["github_issue"] = GitHubIssueTool(
        github_token=env_config.github_token,
        github_repository=env_config.github_repository,
    )
    return tools

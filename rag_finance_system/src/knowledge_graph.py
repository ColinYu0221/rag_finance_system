"""
knowledge_graph.py
Neo4j 知识图谱存储层 — 节点/边 CRUD + 图谱遍历查询。
显式关系（正则提取），不依赖 LLM。
"""
import os
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class KnowledgeGraph:
    """Neo4j 金融法规知识图谱。

    节点: Document / Article / Term / Authority
    边:   BELONGS_TO / REFERENCES / DEFINES / MENTIONS / ISSUED_BY / RELATES_TO
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._connected = False
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            self._connected = True
            self._ensure_constraints()
            logger.info(f"Neo4j 已连接: {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j 不可用，图谱功能将跳过: {e}")
            self._driver = None
            self._connected = False

    def _run(self, query: str, params: Optional[dict] = None) -> List[dict]:
        if not self._connected or self._driver is None:
            return []
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"Neo4j 查询失败: {e}")
            return []

    # ── 约束 ──

    def _ensure_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Authority) REQUIRE a.name IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                self._run(cypher)
            except Exception:
                pass
        self._ensure_indexes()

    # ── 索引 ──

    def _ensure_indexes(self):
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.law_name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.article_num)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX IF NOT EXISTS FOR (au:Authority) ON (au.short)",
        ]
        for cypher in indexes:
            try:
                self._run(cypher)
            except Exception:
                pass

    # ── 清空 ──

    def clear(self):
        if self._connected:
            self._run("MATCH (n) DETACH DELETE n")
            logger.info("知识图谱已清空")

    # ── 节点写入 (MERGE 幂等) ──

    def upsert_document(self, name: str, doc_type: str = "law", source: str = "") -> bool:
        if not name:
            return False
        self._run(
            """
            MERGE (d:Document {name: $name})
            SET d.doc_type = $doc_type, d.source = $source
            """,
            {"name": name, "doc_type": doc_type, "source": source},
        )
        return True

    def upsert_article(
        self,
        chunk_id: str,
        text: str,
        article_num: str = "",
        law_name: str = "",
        source: str = "",
    ) -> bool:
        if not chunk_id:
            return False
        self._run(
            """
            MERGE (a:Article {chunk_id: $chunk_id})
            SET a.text = $text, a.article_num = $article_num,
                a.law_name = $law_name, a.source = $source
            """,
            {
                "chunk_id": chunk_id,
                "text": text[:4096],
                "article_num": article_num,
                "law_name": law_name,
                "source": source,
            },
        )
        return True

    def upsert_term(self, name: str, definition: str = "", category: str = "") -> bool:
        if not name:
            return False
        self._run(
            """
            MERGE (t:Term {name: $name})
            SET t.definition = $definition, t.category = $category
            """,
            {"name": name, "definition": definition[:1024], "category": category},
        )
        return True

    def upsert_authority(self, name: str, short: str = "") -> bool:
        if not name:
            return False
        self._run(
            """
            MERGE (a:Authority {name: $name})
            SET a.short = $short
            """,
            {"name": name, "short": short},
        )
        return True

    # ── 边写入 (MERGE 幂等) ──

    def link_article_to_document(self, chunk_id: str, doc_name: str) -> bool:
        if not chunk_id or not doc_name:
            return False
        self._run(
            """
            MATCH (a:Article {chunk_id: $chunk_id})
            MATCH (d:Document {name: $doc_name})
            MERGE (a)-[r:BELONGS_TO]->(d)
            """,
            {"chunk_id": chunk_id, "doc_name": doc_name},
        )
        return True

    def link_article_reference(self, from_chunk_id: str, to_law: str, to_article: str) -> bool:
        """建立条文引用边 from --[REFERENCES]-> to，其中 to 通过 law_name+article_num 匹配。"""
        if not from_chunk_id or not to_law:
            return False
        self._run(
            """
            MATCH (from:Article {chunk_id: $from_id})
            MATCH (to:Article {law_name: $to_law, article_num: $to_article})
            MERGE (from)-[r:REFERENCES {target_law: $to_law, target_article: $to_article}]->(to)
            """,
            {
                "from_id": from_chunk_id,
                "to_law": to_law,
                "to_article": to_article,
            },
        )
        return True

    def link_term_definition(self, chunk_id: str, term_name: str) -> bool:
        """条文定义了某个术语."""
        if not chunk_id or not term_name:
            return False
        self._run(
            """
            MATCH (a:Article {chunk_id: $chunk_id})
            MATCH (t:Term {name: $term_name})
            MERGE (a)-[r:DEFINES]->(t)
            """,
            {"chunk_id": chunk_id, "term_name": term_name},
        )
        return True

    def link_term_mention(self, chunk_id: str, term_name: str) -> bool:
        """条文提及某个术语."""
        if not chunk_id or not term_name:
            return False
        self._run(
            """
            MATCH (a:Article {chunk_id: $chunk_id})
            MATCH (t:Term {name: $term_name})
            MERGE (a)-[r:MENTIONS]->(t)
            """,
            {"chunk_id": chunk_id, "term_name": term_name},
        )
        return True

    def link_document_authority(self, doc_name: str, authority_name: str) -> bool:
        """文档由监管机构发布或关联."""
        if not doc_name or not authority_name:
            return False
        self._run(
            """
            MATCH (d:Document {name: $doc_name})
            MATCH (a:Authority {name: $authority_name})
            MERGE (d)-[r:ISSUED_BY]->(a)
            """,
            {"doc_name": doc_name, "authority_name": authority_name},
        )
        return True

    def link_document_relation(
        self,
        from_doc: str,
        to_doc: str,
        relation_type: str = "related",
    ) -> bool:
        """文档级法规关系，如上位法/依据/配套/适用."""
        if not from_doc or not to_doc:
            return False
        self._run(
            """
            MATCH (from:Document {name: $from_doc})
            MATCH (to:Document {name: $to_doc})
            MERGE (from)-[r:RELATES_TO {relation_type: $relation_type}]->(to)
            """,
            {
                "from_doc": from_doc,
                "to_doc": to_doc,
                "relation_type": relation_type,
            },
        )
        return True


    # ── 批量节点写入 ──

    def upsert_articles_batch(self, chunks: List[Dict[str, Any]]) -> int:
        count = 0
        for c in chunks:
            if self.upsert_article(
                chunk_id=c.get("chunk_id", ""),
                text=c.get("text", ""),
                article_num=c.get("article_num", ""),
                law_name=c.get("law_name", ""),
                source=c.get("source", ""),
            ):
                count += 1
        return count

    def upsert_documents_batch(self, chunks: List[Dict[str, Any]]) -> int:
        seen: set = set()
        count = 0
        for c in chunks:
            doc_name = c.get("law_name", "") or c.get("source", "")
            if not doc_name or doc_name in seen:
                continue
            seen.add(doc_name)
            if self.upsert_document(
                name=doc_name,
                doc_type=c.get("doc_type", "law"),
                source=c.get("source", ""),
            ):
                count += 1
        return count

    def link_articles_to_documents_batch(self, chunks: List[Dict[str, Any]]) -> int:
        count = 0
        for c in chunks:
            doc_name = c.get("law_name", "") or c.get("source", "")
            if self.link_article_to_document(c.get("chunk_id", ""), doc_name):
                count += 1
        return count

    # ── 图谱查询 ──

    def get_related_articles(
        self,
        law_names: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
        article_nums: Optional[List[str]] = None,
        authorities: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """根据命中的法律名/术语/条文号/机构，查询图谱中关联的条文。

        返回: [{text, chunk_id, law_name, article_num, source, relation, path_length}]
        """
        results: List[Dict[str, Any]] = []

        if law_names:
            records = self._run(
                """
                MATCH (a:Article)-[:BELONGS_TO]->(d:Document)
                WHERE d.name IN $law_names
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'belongs_to' AS relation, 1 AS path_length
                LIMIT $max_results
                """,
                {"law_names": law_names, "max_results": max_results},
            )
            results.extend(records)

            related_doc_records = self._run(
                """
                MATCH (src:Document)-[rel:RELATES_TO]->(dst:Document)
                WHERE src.name IN $law_names
                MATCH (a:Article)-[:BELONGS_TO]->(dst)
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'related_document_' + rel.relation_type AS relation, 2 AS path_length
                LIMIT $max_results
                """,
                {"law_names": law_names, "max_results": max_results},
            )
            results.extend(related_doc_records)

        if terms:
            records = self._run(
                """
                MATCH (a:Article)-[:DEFINES]->(t:Term)
                WHERE t.name IN $terms
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'defines' AS relation, 1 AS path_length
                LIMIT $max_results
                """,
                {"terms": terms, "max_results": max_results},
            )
            results.extend(records)

            mention_records = self._run(
                """
                MATCH (a:Article)-[:MENTIONS]->(t:Term)
                WHERE t.name IN $terms
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'mentions' AS relation, 1 AS path_length
                LIMIT $max_results
                """,
                {"terms": terms, "max_results": max_results},
            )
            results.extend(mention_records)

            ref_records = self._run(
                """
                MATCH (a:Article)-[:REFERENCES]->(b:Article)
                MATCH (b)-[:DEFINES]->(t:Term)
                WHERE t.name IN $terms
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'references_term_definer' AS relation, 2 AS path_length
                LIMIT $max_results
                """,
                {"terms": terms, "max_results": max_results},
            )
            results.extend(ref_records)

        if article_nums:
            records = self._run(
                """
                MATCH (a:Article)-[:REFERENCES]->(b:Article)
                WHERE b.article_num IN $article_nums
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'references_article' AS relation, 1 AS path_length
                LIMIT $max_results
                """,
                {"article_nums": article_nums, "max_results": max_results},
            )
            results.extend(records)

            ref_by_records = self._run(
                """
                MATCH (a:Article)
                WHERE a.article_num IN $article_nums
                MATCH (a)<-[:REFERENCES]-(b:Article)
                RETURN b.text AS text, b.chunk_id AS chunk_id,
                       b.law_name AS law_name,
                       b.article_num AS article_num, b.source AS source,
                       'referenced_by' AS relation, 1 AS path_length
                LIMIT $max_results
                """,
                {"article_nums": article_nums, "max_results": max_results},
            )
            results.extend(ref_by_records)

        if authorities:
            records = self._run(
                """
                MATCH (d:Document)-[:ISSUED_BY]->(au:Authority)
                WHERE au.name IN $authorities
                MATCH (a:Article)-[:BELONGS_TO]->(d)
                RETURN a.text AS text, a.chunk_id AS chunk_id,
                       a.law_name AS law_name,
                       a.article_num AS article_num, a.source AS source,
                       'issued_by_authority' AS relation, 2 AS path_length
                LIMIT $max_results
                """,
                {"authorities": authorities, "max_results": max_results},
            )
            results.extend(records)

        deduped: Dict[str, dict] = {}
        for r in results:
            key = r.get("chunk_id") or r.get("text", "")[:60]
            existing = deduped.get(key)
            if existing is None or r.get("path_length", 99) < existing.get("path_length", 99):
                deduped[key] = r
        return list(deduped.values())[:max_results]

    def get_graph_facts(
        self,
        law_names: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
        authorities: Optional[List[str]] = None,
        max_results: int = 8,
    ) -> List[str]:
        if not self._connected:
            return []

        facts: List[str] = []
        seen: set[str] = set()

        if law_names:
            records = self._run(
                """
                MATCH (d:Document)-[rel:RELATES_TO]->(other:Document)
                WHERE d.name IN $law_names
                RETURN d.name AS doc, rel.relation_type AS relation_type, other.name AS other
                LIMIT $max_results
                """,
                {"law_names": law_names, "max_results": max_results},
            )
            for rec in records:
                fact = f"法规关系: {rec['doc']} --{rec['relation_type']}--> {rec['other']}"
                if fact not in seen:
                    seen.add(fact)
                    facts.append(fact)

        if authorities:
            records = self._run(
                """
                MATCH (d:Document)-[:ISSUED_BY]->(a:Authority)
                WHERE a.name IN $authorities
                RETURN d.name AS doc, a.name AS authority
                LIMIT $max_results
                """,
                {"authorities": authorities, "max_results": max_results},
            )
            for rec in records:
                fact = f"发布关系: {rec['authority']} 发布或关联 {rec['doc']}"
                if fact not in seen:
                    seen.add(fact)
                    facts.append(fact)

        if terms:
            records = self._run(
                """
                MATCH (a:Article)-[:DEFINES|MENTIONS]->(t:Term)
                WHERE t.name IN $terms
                RETURN a.law_name AS law_name, a.article_num AS article_num, t.name AS term
                LIMIT $max_results
                """,
                {"terms": terms, "max_results": max_results},
            )
            for rec in records:
                article_label = rec['article_num'] or '未标注条号'
                fact = f"术语关系: {rec['term']} 出现在 {rec['law_name']} 第{article_label}条"
                if fact not in seen:
                    seen.add(fact)
                    facts.append(fact)

        return facts[:max_results]

    def get_article_relations(self, law_name: str, article_num: str) -> Dict[str, Any]:
        """查询指定条文的引用关系、所属文档、关联法规。"""
        if not self._connected:
            return {}
        result: Dict[str, Any] = {}

        target_records = self._run(
            """
            MATCH (a:Article)
            WHERE a.law_name = $law_name AND a.article_num = $article_num
            RETURN a.text AS text, a.law_name AS law_name,
                   a.article_num AS article_num, a.source AS source,
                   a.chunk_id AS chunk_id
            LIMIT 1
            """,
            {"law_name": law_name, "article_num": article_num},
        )
        if not target_records:
            return {}
        target = target_records[0]
        result["target"] = {
            "law_name": target["law_name"],
            "article_num": target["article_num"],
            "text": target["text"],
            "source": target["source"],
        }

        incoming = self._run(
            """
            MATCH (src:Article)-[:REFERENCES]->(dst:Article)
            WHERE dst.law_name = $law_name AND dst.article_num = $article_num
            RETURN src.law_name AS law_name, src.article_num AS article_num,
                   src.text AS text, src.source AS source
            LIMIT 20
            """,
            {"law_name": law_name, "article_num": article_num},
        )
        result["incoming_refs"] = incoming

        outgoing = self._run(
            """
            MATCH (src:Article)-[:REFERENCES]->(dst:Article)
            WHERE src.law_name = $law_name AND src.article_num = $article_num
            RETURN dst.law_name AS law_name, dst.article_num AS article_num,
                   dst.text AS text, dst.source AS source
            LIMIT 20
            """,
            {"law_name": law_name, "article_num": article_num},
        )
        result["outgoing_refs"] = outgoing

        parent_records = self._run(
            """
            MATCH (a:Article)-[:BELONGS_TO]->(d:Document)
            WHERE a.law_name = $law_name AND a.article_num = $article_num
            RETURN d.name AS name, d.doc_type AS doc_type, d.source AS source
            LIMIT 1
            """,
            {"law_name": law_name, "article_num": article_num},
        )
        if parent_records:
            result["parent_document"] = parent_records[0]

        related_docs = self._run(
            """
            MATCH (src:Document)-[rel:RELATES_TO]->(dst:Document)
            WHERE src.name = $law_name
            RETURN dst.name AS name, rel.relation_type AS relation_type, 'outgoing' AS direction
            LIMIT 10
            """,
            {"law_name": law_name},
        )
        related_docs += self._run(
            """
            MATCH (src:Document)-[rel:RELATES_TO]->(dst:Document)
            WHERE dst.name = $law_name
            RETURN src.name AS name, rel.relation_type AS relation_type, 'incoming' AS direction
            LIMIT 10
            """,
            {"law_name": law_name},
        )
        result["related_documents"] = related_docs

        related_articles = []
        for rd in related_docs[:3]:
            ra_records = self._run(
                """
                MATCH (a:Article)-[:BELONGS_TO]->(d:Document)
                WHERE d.name = $doc_name
                RETURN a.law_name AS law_name, a.article_num AS article_num,
                       a.text AS text, a.source AS source
                LIMIT 2
                """,
                {"doc_name": rd["name"]},
            )
            related_articles.extend(ra_records)
        result["related_articles"] = related_articles

        return result

    def stats(self) -> Dict[str, int]:
        if not self._connected:
            return {}
        labels = ["Document", "Article", "Term", "Authority"]
        result: Dict[str, int] = {}
        for label in labels:
            records = self._run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            result[label.lower()] = records[0]["cnt"] if records else 0
        rel_records = self._run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt"
        )
        for rec in rel_records:
            result[f"rel_{rec['t'].lower()}"] = rec["cnt"]
        return result
